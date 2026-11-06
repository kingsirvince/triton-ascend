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

#ifndef TRITON_ADAPTER_CONVERSION_LOWERADDRESSVALUEPOINTERS_H
#define TRITON_ADAPTER_CONVERSION_LOWERADDRESSVALUEPOINTERS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

// Rewrite the "address value" sub-graph of pointer tensors produced by
// tt.int_to_ptr into plain i64 tensors when the address values only take part
// in data flow (addptr / select / broadcast / reshape / compare / ptr_to_int).
//
// A pointer tensor that (transitively) feeds a memory access or escapes across
// a region/function boundary keeps its current representation; in that case the
// offending operation is reported and the pass fails, because the backend has
// no memref form for a tensor of runtime addresses.
LogicalResult lowerAddressValuePointerTensors(ModuleOp moduleOp);

// Same rewriting exposed as a pass, so that it can run early in the TTIR
// pipeline (before the unstructure/memory stages).
std::unique_ptr<Pass> createLowerAddressValuePointersPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_CONVERSION_LOWERADDRESSVALUEPOINTERS_H
