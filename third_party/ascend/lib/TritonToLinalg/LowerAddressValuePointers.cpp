/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Address-value pointer tensors: tensors of !tt.ptr<T> whose value is a runtime
// byte address (produced by tt.int_to_ptr) rather than the base address of a
// known allocation. Such values cannot be described by a memref descriptor, so
// they are lowered to plain i64 tensors as long as they stay in data flow.
//
// The rewrite is deliberately conservative: as soon as an address value (or any
// value of its producer chain) feeds a memory access, an argument, a callee or a
// region boundary, the whole connected component keeps its current
// representation and the existing pointer-carrier path is used unchanged.

#include "TritonToLinalg/LowerAddressValuePointers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace mlir {
namespace triton {
namespace {

// A "plain" pointer is !tt.ptr<T> with a scalar pointee. Block pointers
// (!tt.ptr<tensor<...>>) are handled by the descriptor path and are excluded.
bool isPlainPointer(Type type) {
  if (auto pointerType = dyn_cast<triton::PointerType>(type))
    return !isa<ShapedType>(pointerType.getPointeeType());
  if (auto shapedType = dyn_cast<ShapedType>(type))
    if (auto pointerType = dyn_cast<triton::PointerType>(
            shapedType.getElementType()))
      return !isa<ShapedType>(pointerType.getPointeeType());
  return false;
}

bool isShapeOnly(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && isPlainPointer(shapedType.getElementType());
}

// Pointer type carried by a scalar or shaped pointer value.
triton::PointerType getPointerElementType(Type type) {
  if (auto pointerType = dyn_cast<triton::PointerType>(type))
    return pointerType;
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return dyn_cast<triton::PointerType>(shapedType.getElementType());
  return nullptr;
}

// The integer type that carries an address value of the given pointer type.
Type getAddressType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.clone(IntegerType::get(type.getContext(), 64));
  return IntegerType::get(type.getContext(), 64);
}

// Ops that transport an address value without dereferencing it.
bool isAddressPropagator(Operation *op) {
  return isa<arith::SelectOp>(op) || isa<triton::AddPtrOp>(op) ||
         isa<triton::SplatOp>(op) || isa<triton::BroadcastOp>(op) ||
         isa<triton::ExpandDimsOp>(op) || isa<triton::ReshapeOp>(op) ||
         isa<triton::BitcastOp>(op);
}

// Comparing addresses is an integer comparison. Ordered pointer comparisons
// are meaningless and are not rewritten.
bool isAddressComparison(Operation *op) {
  auto cmpOp = dyn_cast<arith::CmpIOp>(op);
  if (!cmpOp)
    return false;
  return cmpOp.getPredicate() == arith::CmpIPredicate::eq ||
         cmpOp.getPredicate() == arith::CmpIPredicate::ne;
}

// Byte size of the pointee, needed to translate addptr element offsets into
// byte offsets.
std::optional<int64_t> getPointeeByteSize(Type pointeeType) {
  if (!pointeeType.isIntOrFloat())
    return std::nullopt;
  int64_t bitWidth = pointeeType.getIntOrFloatBitWidth();
  return std::max<int64_t>(1, bitWidth / 8);
}

class AddressValueLowering {
public:
  explicit AddressValueLowering(ModuleOp moduleOp) : module(moduleOp) {}

  bool run();

private:
  void collectSeeds();
  void computeCandidates();
  void computeBlocked();
  void propagateBlockedBackwards();
  void rewriteInProgramOrder();
  LogicalResult diagnoseUnsupportedUses();

  Value lookup(Value value) const {
    auto it = replacements.find(value);
    return it == replacements.end() ? value : it->second;
  }

  bool isRewritable(Value value) const {
    return candidates.contains(value) && !blocked.contains(value);
  }

  // True when every pointer operand of the op is an address value, i.e. the op
  // can be reproduced in the integer domain without mixing provenances.
  bool operandsAreAddresses(Operation *op) const {
    for (Value operand : op->getOperands()) {
      if (!isPlainPointer(operand.getType()))
        continue;
      if (!candidates.contains(operand))
        return false;
    }
    return true;
  }

  LogicalResult rewriteOp(Operation *op);

  ModuleOp module;

  // Results of tt.int_to_ptr that are tensors of pointers: the roots of every
  // address-value data flow.
  SmallVector<Value> seeds;
  // Values that can be described as an i64 address (forward closure of seeds).
  DenseSet<Value> candidates;
  // Values that must keep their pointer representation.
  DenseSet<Value> blocked;
  // Old address value -> new i64 value.
  DenseMap<Value, Value> replacements;
  SmallVector<Operation *> deadOps;
};

void AddressValueLowering::collectSeeds() {
  module.walk([&](triton::IntToPtrOp op) {
    if (isShapeOnly(op.getType()))
      seeds.push_back(op.getResult());
  });
}

void AddressValueLowering::computeCandidates() {
  SmallVector<Value> worklist(seeds);
  candidates.insert(seeds.begin(), seeds.end());

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (!isAddressPropagator(user) || user->getNumResults() != 1)
        continue;
      // A propagator mixing an address value with a pointer of other
      // provenance (a real allocation, a function argument, ...) cannot be
      // expressed in the integer domain: leave the whole op alone.
      if (!operandsAreAddresses(user))
        continue;

      Value result = user->getResult(0);
      if (!isPlainPointer(result.getType()))
        continue;
      if (candidates.insert(result).second)
        worklist.push_back(result);
    }
  }
}

void AddressValueLowering::computeBlocked() {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Value value : candidates) {
      if (blocked.contains(value))
        continue;
      for (Operation *user : value.getUsers()) {
        if (isa<triton::PtrToIntOp>(user))
          continue; // address -> integer needs no pointer representation
        if (isAddressPropagator(user) && user->getNumResults() == 1 &&
            candidates.contains(user->getResult(0)))
          continue;
        // Any other consumer (load/store/atomic/call/return/yield/...) forces
        // this value, and therefore its whole producer chain, to stay a
        // pointer.
        blocked.insert(value);
        changed = true;
        break;
      }
    }
  }
}

void AddressValueLowering::propagateBlockedBackwards() {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Value> toBlock;
    for (Value value : candidates) {
      if (blocked.contains(value))
        continue;
      for (Operation *user : value.getUsers()) {
        if (isAddressPropagator(user) && user->getNumResults() == 1 &&
            blocked.contains(user->getResult(0)))
          toBlock.push_back(value);
        // A select/pick that mixes an address value with a real pointer keeps
        // every operand in the pointer domain.
        if (isAddressPropagator(user) && user->getNumResults() == 1 &&
            !candidates.contains(user->getResult(0)))
          toBlock.push_back(value);
      }
    }
    for (Value value : toBlock) {
      if (blocked.insert(value).second)
        changed = true;
    }
  }
}

LogicalResult AddressValueLowering::rewriteOp(Operation *op) {
  // int_to_ptr(tensor<i64>) -> tensor<!tt.ptr<T>>: the address is already the
  // integer operand, so the op disappears.
  if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
    if (!isRewritable(intToPtrOp.getResult()))
      return success();
    Value address = lookup(intToPtrOp.getSrc());
    Type wanted = getAddressType(intToPtrOp.getResult().getType());
    if (address.getType() != wanted) {
      if (auto shapedType = dyn_cast<ShapedType>(address.getType()))
        address = OpBuilder(op).create<arith::ExtSIOp>(op->getLoc(), wanted,
                                                      address);
      else
        return op->emitError("unexpected integer type for int_to_ptr source");
    }
    replacements[intToPtrOp.getResult()] = address;
    deadOps.push_back(op);
    return success();
  }

  // ptr_to_int of an address value is an identity in the integer domain.
  if (auto ptrToIntOp = dyn_cast<triton::PtrToIntOp>(op)) {
    if (!isRewritable(ptrToIntOp.getSrc()))
      return success();
    Value address = lookup(ptrToIntOp.getSrc());
    if (address.getType() != ptrToIntOp.getResult().getType())
      return op->emitError("address type mismatch in ptr_to_int lowering");
    ptrToIntOp.getResult().replaceAllUsesWith(address);
    deadOps.push_back(op);
    return success();
  }

  // addptr on an address value is a byte-level addition.
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    if (!isRewritable(addPtrOp.getResult()))
      return success();
    triton::PointerType resultPointerType =
        getPointerElementType(addPtrOp.getResult().getType());
    if (!resultPointerType)
      return op->emitError("addptr on an address value must produce a pointer");
    std::optional<int64_t> elementSize =
        getPointeeByteSize(resultPointerType.getPointeeType());
    if (!elementSize)
      return op->emitError("unsupported pointee type in addptr lowering");

    Location loc = op->getLoc();
    OpBuilder builder(op);
    Value base = lookup(addPtrOp.getPtr());
    Value offsets = lookup(addPtrOp.getOffset());
    Type i64Type = builder.getI64Type();

    if (auto shapedOffsetType = dyn_cast<ShapedType>(offsets.getType())) {
      if (!shapedOffsetType.getElementType().isInteger(64)) {
        offsets = builder.create<arith::ExtSIOp>(
            loc, shapedOffsetType.clone(i64Type), offsets);
        shapedOffsetType = cast<ShapedType>(offsets.getType());
      }
    } else if (!offsets.getType().isInteger(64)) {
      offsets = builder.create<arith::ExtSIOp>(loc, i64Type, offsets);
    }

    // A scalar offset applied to a tensor of addresses is a broadcast.
    if (isa<ShapedType>(base.getType()) && !isa<ShapedType>(offsets.getType()))
      offsets =
          builder.create<triton::SplatOp>(loc, base.getType(), offsets);

    Value byteOffsets = offsets;
    if (*elementSize != 1) {
      Value size = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(i64Type, *elementSize));
      if (isa<ShapedType>(offsets.getType()))
        size = builder.create<triton::SplatOp>(loc, offsets.getType(), size);
      byteOffsets = builder.create<arith::MulIOp>(loc, offsets, size);
    }

    Value address = builder.create<arith::AddIOp>(loc, base, byteOffsets);
    replacements[addPtrOp.getResult()] = address;
    deadOps.push_back(op);
    return success();
  }

  // Comparing addresses is an integer comparison.
  if (isAddressComparison(op) && operandsAreAddresses(op)) {
    OpBuilder builder(op);
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      operands.push_back(lookup(operand));
    Operation *newOp = builder.create<arith::CmpIOp>(
        op->getLoc(), op->getResult(0).getType(),
        cast<arith::CmpIOp>(op).getPredicate(), operands[0], operands[1]);
    replacements[op->getResult(0)] = newOp->getResult(0);
    deadOps.push_back(op);
    return success();
  }

  // Comparing addresses is an integer comparison.
  if (isAddressComparison(op) && operandsAreAddresses(op)) {
    OpBuilder builder(op);
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      operands.push_back(lookup(operand));
    Operation *newOp = builder.create<arith::CmpIOp>(
        op->getLoc(), op->getResult(0).getType(),
        cast<arith::CmpIOp>(op).getPredicate(), operands[0], operands[1]);
    replacements[op->getResult(0)] = newOp->getResult(0);
    deadOps.push_back(op);
    return success();
  }

  // Comparing addresses is an integer comparison.
  if (isAddressComparison(op) && operandsAreAddresses(op)) {
    OpBuilder builder(op);
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      operands.push_back(lookup(operand));
    Operation *newOp = builder.create<arith::CmpIOp>(
        op->getLoc(), op->getResult(0).getType(),
        cast<arith::CmpIOp>(op).getPredicate(), operands[0], operands[1]);
    replacements[op->getResult(0)] = newOp->getResult(0);
    deadOps.push_back(op);
    return success();
  }

  // A bitcast between pointer types is the identity on the byte address.
  if (auto bitcastOp = dyn_cast<triton::BitcastOp>(op)) {
    if (!isRewritable(bitcastOp.getResult()))
      return success();
    Value address = lookup(bitcastOp.getSrc());
    if (address.getType() != getAddressType(bitcastOp.getResult().getType()))
      return op->emitError("address type mismatch in pointer bitcast lowering");
    bitcastOp.getResult().replaceAllUsesWith(address);
    deadOps.push_back(op);
    return success();
  }

  // Shape-only propagators (select / splat / broadcast / expand_dims / reshape)
  // keep their semantics, only the element type changes.
  if (op->getNumResults() == 1 && isRewritable(op->getResult(0))) {
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      operands.push_back(lookup(operand));

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (Value result : op->getResults())
      resultTypes.push_back(getAddressType(result.getType()));

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = Operation::create(state);
    OpBuilder(op).insert(newOp);

    for (size_t i = 0, e = op->getNumResults(); i < e; ++i)
      replacements[op->getResult(i)] = newOp->getResult(i);
    deadOps.push_back(op);
  }
  return success();
}

void AddressValueLowering::rewriteInProgramOrder() {
  SmallVector<Operation *> orderedOps;
  module.walk([&](Operation *op) { orderedOps.push_back(op); });

  for (Operation *op : orderedOps) {
    if (op->getNumResults() == 0)
      continue;
    if (!isRewritable(op->getResult(0)) &&
        !(isa<triton::PtrToIntOp>(op) && isRewritable(op->getOperand(0))))
      continue;
    (void)rewriteOp(op);
  }

  // Everything that was rewritten is now dead: its remaining users were
  // rewritten as well (guaranteed by the blocked-component analysis).
  for (Operation *op : llvm::reverse(deadOps)) {
    if (op->use_empty())
      op->erase();
  }
}

// Address-value tensors that cannot be dropped into the integer domain stay in
// the pointer domain, which the backend cannot legalize. Report why, instead of
// letting the generic legalization error (or a downstream assertion) surface.
LogicalResult AddressValueLowering::diagnoseUnsupportedUses() {
  bool failed = false;
  for (Value value : candidates) {
    if (!blocked.contains(value))
      continue;
    for (Operation *user : value.getUsers()) {
      if (isa<triton::PtrToIntOp>(user))
        continue;
      if (isAddressComparison(user) && operandsAreAddresses(user))
        continue;
      if (isAddressPropagator(user)) {
        // Either the propagator feeds the actual problem further down, or it
        // mixes provenances and cannot be rewritten at all.
        if (user->getNumResults() == 1 &&
            candidates.contains(user->getResult(0)))
          continue;
        user->emitError("address-value pointer tensor is mixed with a pointer "
                        "that does not come from tt.int_to_ptr; such a value "
                        "has no memref representation on Ascend");
        failed = true;
        continue;
      }
      if (isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
              triton::AtomicCASOp>(user) &&
          user->getOperand(0) == value) {
        user->emitError(
            "address-value pointer tensor is used as the base of a memory "
            "access; on Ascend every lane of a load/store must share one base "
            "pointer, so a tensor of tt.int_to_ptr addresses cannot be "
            "dereferenced yet");
        failed = true;
        continue;
      }
      user->emitError("address-value pointer tensor escapes the integer domain "
                      "here (region yield, function return, call argument or "
                      "unsupported operation); it cannot be represented on "
                      "Ascend");
      failed = true;
    }
  }
  return failure(failed);
}

bool AddressValueLowering::run() {
  collectSeeds();
  if (seeds.empty())
    return true;

  computeCandidates();
  computeBlocked();
  propagateBlockedBackwards();

  // Report before rewriting so that a rejected module keeps a readable IR.
  if (failed(diagnoseUnsupportedUses()))
    return false;

  rewriteInProgramOrder();
  return true;
}

} // namespace

LogicalResult lowerAddressValuePointerTensors(ModuleOp moduleOp) {
  AddressValueLowering lowering(moduleOp);
  if (!lowering.run())
    return failure();
  return success();
}

namespace {
// Pipeline entry point. The rewriting has to run before the unstructure and
// memory stages, which assume every pointer value is derived from one base
// pointer; those stages cannot even report such a value cleanly.
class LowerAddressValuePointersPass
    : public PassWrapper<LowerAddressValuePointersPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAddressValuePointersPass)

  StringRef getArgument() const final { return "lower-address-value-pointers"; }
  StringRef getDescription() const final {
    return "Lower tensors of tt.int_to_ptr addresses that only take part in "
           "data flow to i64 tensors";
  }

  void runOnOperation() override {
    if (failed(lowerAddressValuePointerTensors(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerAddressValuePointersPass() {
  return std::make_unique<LowerAddressValuePointersPass>();
}

} // namespace triton
} // namespace mlir
