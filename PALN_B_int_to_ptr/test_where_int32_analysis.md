# test_where[1-\*int32] 编译失败分析与解决方案

- 分支：`1104_main-dev_test_where_int32`（`/home/w30038269/code/triton-ascend`，当前安装的 triton-ascend 即由该路径编译安装）
- 环境：Triton-Ascend 3.2.1 / triton 3.6.0 / CANN 9.0.0 / torch_npu 2.10.0.post2 / Ascend910B4
- 现象：`test_core.py::test_where[1-*int32]` 在 `ConvertTritonIRToLinalgIR` 阶段编译失败

---

## 1. 结论（TL;DR）

**是的，根因是 triton-ascend 目前不支持“结果类型为 tensor 的 `tt.int_to_ptr`”。**

- triton-ascend 支持 **标量** `int_to_ptr`（`i64 -> !tt.ptr<T>`），它被 lowering 成“不透明指针载体”（memref 描述符）。
- 但对 **tensor 形态** `tensor<Nxi64> -> tensor<Nx!tt.ptr<T>>`，`IntToPtrConverter` 第一行就 `notifyMatchFailure` 直接放弃匹配，而 `tt.int_to_ptr` 又因为结果类型含指针被 Triton 方言的动态合法性判为 illegal，于是报
  `failed to legalize operation 'tt.int_to_ptr' that was explicitly marked illegal`。

**不是** `tl.where` / pointer-select 的问题：探针实验证明“由 `addptr` 产生的指针张量做 select”“标量指针 select”在当前后端都 **正常工作**；只有“指针张量来自 `int_to_ptr`”这条链路没有任何 lowering 路径。

**推荐修复（方案 A）**：在 `TritonToStructured` 的最前置规范化阶段（`applyPatternsGreedily`）增加两条指针往返消除规则，把
`ptr_to_int(select(c, int_to_ptr(a), int_to_ptr(b)))` 规范化为 `select(c, a, b)`（纯整数），该形态当前后端完全支持。
这一做法与后端已有的“指针跨控制流时用 i64 地址承载”的既有设计（`getIfResultCarrierType` / `PromotePointerIterArgsPattern`）一致，改动小、风险可控。

---

## 2. 复现与证据

### 2.1 运行方式

```bash
docker exec -it w30038269 bash
source /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/bin/setenv.bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH=/home/w30038269/env/NPUIR/3.2.2_post2_1010/bishengir/bin:$PATH
cd /tmp   # 必须离开 triton 源码树，否则 import 到未编译的源码树
python -m pytest <lang-triton>/python/test/unit/language/test_core.py -k "test_where" -x
```

### 2.2 报错原文

```
.../test_core.py:813: error: failed to legalize operation 'tt.int_to_ptr' that was explicitly marked illegal
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
                            ^
.../test_core.py:813: error: failed to convert memory operations
note: Pipeline failed while executing [`TritonToLinalg` on 'builtin.module' operation]
```

### 2.3 失败时的 TTIR（与用例一一对应）

```mlir
%6   = tt.int_to_ptr %3  : tensor<64xi64> -> tensor<64x!tt.ptr<i32>>   // ← 非法点
%11  = tt.int_to_ptr %10 : tensor<64xi64> -> tensor<64x!tt.ptr<i32>>
%16  = arith.select %15, %7, %11 : tensor<64xi1>, tensor<64x!tt.ptr<i32>>
%17  = tt.ptr_to_int %16 : tensor<64x!tt.ptr<i32>> -> tensor<64xi64>
tt.store %13, %17 : tensor<64x!tt.ptr<i64>>
```

关键点：`arith.select` 与 `tt.store` 本身都能被后端处理（见第 4 节能力矩阵），真正无法 legalize 的只有 `tt.int_to_ptr`（tensor 结果）。

### 2.4 实际 pass 流水线

```
builtin.module(
  convert-triton-distributed-to-hivm, triton-control-flow-opt, triton-to-structured,
  discrete-mask-access-conversion, triton-to-annotation, triton-to-unstructure,
  triton-to-hivm, triton-to-hfusion, triton-to-llvm, bubble-up-operation,
  triton-to-structured, triton-to-linalg)      // ← 在此 pass 失败
```

`triton-to-structured` / `triton-to-unstructure` 都已跑过，但没有改写这条指针链，说明前端各 pass 均未覆盖该形态。

---

## 3. 根因链路

### 3.1 前端语义：用例为什么要造出 pointer tensor

`test_where` 在 `TEST_POINTERS=True` 分支里：

```python
a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)   # i64 地址张量 -> 指针张量
b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
output = tl.where(decide, a, b)                          # 指针张量 select
tl.store(output_ptr + offsets, tl.ptr_to_int(output), mask=mask)
```

这是上游 Triton（GPU）合法的“指针值张量”用法；在 GPU 后端指针就是一个 64 位整数，`int_to_ptr` / `ptr_to_int` / `select` 都是纯整数运算。

### 3.2 triton-ascend 的指针模型

后端不做“指针=整数”的映射，而是把 Triton 指针降级成 memref 描述符（base/offset/stride，即 BlockData）：

[triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L602-L630](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L602-L630)

```cpp
static MemRefType convertPointerTensorType(TensorType tensorType, triton::PointerType pointerType) {
  Type elementType = normalizeMemoryElementType(pointerType.getPointeeType(), ...);
  return MemRefType::get(tensorType.getShape(), elementType);   // tensor<Nx!tt.ptr<T>> -> memref<NxT>
}
```

因此“指针张量”在后端只有一种合法含义：**一片内存的地址描述**（`memref<NxT>`）。
而 `int_to_ptr` 的含义是“一段运行期地址值变成指针”，它无法映射成 memref（memref 不能承载逐 lane 的任意地址），所以张量形态在这个模型下天然没有对应实现——这是设计层面的空缺，不是某个 pattern 写错。

### 3.3 直接失败点：`IntToPtrConverter` 只接受标量

[triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L3479-L3499](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L3479-L3499)

```cpp
LogicalResult IntToPtrConverter::matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                                                ConversionPatternRewriter &rewriter) const {
  auto pointerType = dyn_cast<triton::PointerType>(op.getType());     // tensor<...> -> nullptr
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return rewriter.notifyMatchFailure(op, "only scalar pointer reconstruction is supported");
  ...
  auto pointerCast = createScalarPointerCast(rewriter, op.getLoc(), resultType, adaptor.getSrc());
  rewriter.replaceOp(op, pointerCast.getResult());
  return success();
}
```

`op.getType()` 是 `tensor<64x!tt.ptr<i32>>`（`TensorType`），`dyn_cast<triton::PointerType>` 返回空 → 直接 match failure。

### 3.4 为什么“没人兜底”：合法性判定把该 op 判为 illegal

[triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L1086-L1092](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L1086-L1092)

```cpp
target.addDynamicallyLegalDialect<triton::TritonDialect, triton::ascend::TritonAscendDialect>(
    [](Operation *op) {
      return hasPointerFreeTypes(op->getOperandTypes()) && hasPointerFreeTypes(op->getResultTypes());
    });
```

`tt.int_to_ptr` 的结果含 `!tt.ptr` → illegal；唯一能改写它的 `IntToPtrConverter` 又拒绝匹配 → 出现在“explicitly marked illegal 且无可用 pattern”的僵局，`applyPartialConversion` 直接报错。

补充两点旁证（都表明现有实现是“标量专用”）：

- [TritonOpConverter.cpp#L118](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L118) `isIntToPtrBasedScalarPointer`：间接 load 路径只在“标量 int_to_ptr”场景下重建指针。
- [BlockPtrAnalysis.cpp#L706](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/BlockPtrAnalysis.cpp#L706)：遇到“指针张量 select”直接 `emitOpError("tensor-of-pointers select must be lowered before BlockData parsing")`，即代码作者预期该形态应在更早阶段被消除，但实际上没有任何 pass 负责消除它。

### 3.5 为什么 `arith.select` / `tt.ptr_to_int` 不是元凶

- [TritonToLinalgPass.cpp#L1113-L1115](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L1113-L1115)：只有**标量**指针 select 被判 illegal（`isScalarPointerSelect`），指针张量 select 被判为 legal → 后端默认它“不需要也没法被转换”，只能靠更早的 pass 把它消化掉。
- [TritonOpConverter.cpp#L3501-L3512](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L3501-L3512) `PtrToIntConverter` 通过 `materializePointerAddress` 取地址，本身也是面向 memref（即真实地址描述）的。

---

## 4. 能力矩阵（实测）

探针脚本：[probe_pointer_casts.py](file:///home/w30038269/test/1104_test_where_int32/probe_pointer_casts.py)（每个 case 独立编译运行）

| # | 形态 | 结果 |
|---|------|------|
| 1 | `tl.where` on `int64` 张量（无指针） | ✅ OK |
| 2 | 标量 `int_to_ptr` → `tl.load` | ✅ OK |
| 3 | 张量 `int_to_ptr` → `ptr_to_int` | ❌ `failed to legalize 'tt.int_to_ptr'` |
| 4 | 张量 `int_to_ptr` → `select` → `ptr_to_int`（本次用例） | ❌ 同上报错 |
| 5 | 两个 `addptr` 指针张量做 `tl.where` → `tl.load` | ✅ OK |
| 6 | 标量指针 `tl.where` → `tl.load` | ✅ OK |

结论：后端对“有 base+offset 出处的指针张量”完全支持（case 5/6），唯独不支持“纯地址值张量”的指针化（case 3/4）。

另外 `test_where` 全量 13 个参数化组合中，只有 `float64` 与 `*int32` 失败；`*int32` 之外的所有整型/浮点组合（包括 `int64`）均通过，说明除指针化这一步外，该用例的其余结构对本后端是可行的。

---

## 5. 解决方案

### 方案 A（推荐）：指针往返消除 + `ptr_to_int` 下沉 select

#### A.1 思路

`test_where` 的指针分支在语义上等价于“对两个 i64 地址做 select”：

```
ptr_to_int(select(c, int_to_ptr(a), int_to_ptr(b)))  ==  select(c, a, b)
```

因为 `int_to_ptr`/`ptr_to_int` 在 64 位地址空间上是恒等映射，且 `select` 逐元素。改写后不再存在任何指针值张量，后端现有能力即可编译（等价于矩阵 case 1，实测通过）。

#### A.2 需要新增的两条规范化

```cpp
// P1: ptr_to_int(int_to_ptr(x)) -> x        （任意 shape，含标量）
// P2: ptr_to_int(select(c, p, q)) -> select(c, ptr_to_int(p), ptr_to_int(q))
//     仅限“指针张量”的 select；标量指针 select 保持现状，交给 PointerSelectConverter
```

P2 生成的两个 `ptr_to_int` 会被 P1 折叠，`applyPatternsGreedily` 会自动收敛到 `select(c, a, b)`。

#### A.3 落点

放在 **`TritonToStructured` 最前置的规范化阶段**（流水线中第 2 个 pass，早于 unstructure/hivm/linalg，且该处已有既有 canonicalization 集合）：

- [TritonToStructuredPass.cpp#L84-L104](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToStructured/TritonToStructuredPass.cpp#L84-L104) `populateTritonToStructuredCanonicalizationPatterns`（`runOnOperation` 内第一步就是 `applyPatternsGreedily`）

备选落点：`TritonToLinalgPass::runOnOperation` 的前置步骤（[TritonToLinalgPass.cpp#L1909-L1927](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp#L1909-L1927) 已有 `inlinePointerCarryingCallees` 这类前置处理），但更靠后，不推荐。

#### A.4 参考实现（patch 级）

头文件 `third_party/ascend/include/TritonToStructured/CannonicalizerConverter.h` 追加：

```cpp
/// ptr_to_int(int_to_ptr(x)) -> x
class PtrToIntOfIntToPtrConverter : public OpRewritePattern<triton::PtrToIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::PtrToIntOp op,
                                PatternRewriter &rewriter) const override;
};

/// ptr_to_int(select(c, p, q)) -> select(c, ptr_to_int(p), ptr_to_int(q))
/// 只处理“指针张量”的 select；标量指针 select 保持原路径。
class PtrToIntSelectSinkConverter : public OpRewritePattern<triton::PtrToIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::PtrToIntOp op,
                                PatternRewriter &rewriter) const override;
};
```

实现（`third_party/ascend/lib/TritonToStructured/CannonicalizerConverter.cpp`）：

```cpp
LogicalResult PtrToIntOfIntToPtrConverter::matchAndRewrite(
    triton::PtrToIntOp op, PatternRewriter &rewriter) const {
  auto intToPtr = op.getSrc().getDefiningOp<triton::IntToPtrOp>();
  if (!intToPtr)
    return failure();
  Value src = intToPtr.getSrc();
  // int_to_ptr 的源宽度即指针宽度（i64），类型一致才可直接替换。
  if (src.getType() != op.getType())
    return failure();
  rewriter.replaceOp(op, src);
  return success();
}

LogicalResult PtrToIntSelectSinkConverter::matchAndRewrite(
    triton::PtrToIntOp op, PatternRewriter &rewriter) const {
  auto selectOp = op.getSrc().getDefiningOp<arith::SelectOp>();
  if (!selectOp)
    return failure();
  auto ptrTensorTy = dyn_cast<RankedTensorType>(selectOp.getType());
  if (!ptrTensorTy || !isa<triton::PointerType>(ptrTensorTy.getElementType()))
    return failure();

  auto loc = op.getLoc();
  auto intTy = rewriter.getIntegerType(64);
  auto intTensorTy = RankedTensorType::get(ptrTensorTy.getShape(), intTy);

  auto toInt = [&](Value ptr) -> Value {
    if (ptr.getType() == intTensorTy)
      return ptr;
    return rewriter.create<triton::PtrToIntOp>(loc, intTensorTy, ptr);
  };
  Value newSelect = rewriter.create<arith::SelectOp>(
      loc, selectOp.getCondition(), toInt(selectOp.getTrueValue()),
      toInt(selectOp.getFalseValue()));
  rewriter.replaceOp(op, newSelect);
  return success();
}
```

注册（`TritonToStructuredPass.cpp` 的 `populateTritonToStructuredCanonicalizationPatterns`）：

```cpp
  patterns.add<CannonicalizerConverter::PtrToIntSelectSinkConverter>(
      patterns.getContext(), PatternBenefit(2));   // 先下沉，再折叠
  patterns.add<CannonicalizerConverter::PtrToIntOfIntToPtrConverter>(
      patterns.getContext());
```

#### A.5 正确性 / 安全性

- 只在“无内存访问语义”的纯数据流上做等价替换：`int_to_ptr`→`ptr_to_int` 往返不改值；`ptr_to_int` 逐元素，可与 `select` 交换次序。
- 不动标量指针 select（`isScalarPointerSelect` 为真的场景），因此不影响 [TritonOpConverter.cpp#L433-L466](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L433-L466) 的既有标量路径与 `test_inttoptr_offset_reset.py` 覆盖的间接 load 场景（间接 load 的 `int_to_ptr` → `tt.load` 链路中不存在 `ptr_to_int` 消费者，规则不触发）。
- 与既有设计一致：后端本就把“指针跨越纯数据流/控制流”表示为 i64 地址（`getIfResultCarrierType` 用 i64 承载 `scf.if` 的指针结果，`PromotePointerIterArgsPattern` 用整数承载循环 iter_args）。方案 A 是把同一原则前移到 tensor 形态。
- 若 `int_to_ptr` 的结果同时被 `tt.load`/`tt.store`（真实访存）使用，P1/P2 不匹配（消费者不是 `ptr_to_int`），行为保持不变——不会把不支持的访存“改坏”，也不会把不支持变支持。

#### A.6 覆盖边界（重要：方案 A 不是“只修这一个用例”，但也不是“修所有 int_to_ptr”）

方案 A 修的是**一整类语义**——“指针值只参与纯数据流，最终回到整数或从不参与访存”，不是针对某个 shape、某个 mask、某个具体用例。判据是 **`int_to_ptr` 结果张量的消费者类型**，而不是测试用例长什么样。

实测覆盖矩阵（探针脚本见 §8，`[OK]/[FAIL]` 为当前分支实测结果，“A 后”为按 pattern 设计应达到的状态）：

| 类别 | 典型写法 | 当前 | 方案 A 后 | 说明 |
|------|----------|------|-----------|------|
| **S** 标量 int_to_ptr | `p = tl.load(ptr).to(tl.pointer_type(t))`，再 `p + offs` | ✅ | ✅ | 现状已支持，A 不涉及（triton-ascend 自带 `test_inttoptr_offset_reset.py` 全为标量形态） |
| **D1** 往返 | `ptr_to_int(int_to_ptr(x))` | ❌ | ✅ | A 的 P1 直接折叠 |
| **D2** 指针值 select → int | `test_where[1-*int32]` 形态 | ❌ | ✅ | P2 + P1 |
| **D3** 多层/嵌套 select | `where(c, p, where(d, p, q))` | ❌ | ✅ | P2 反复应用，greedy 自动收敛 |
| **D4** 指针值跨 `scf.if`（无 addptr） | `if f: p = q` 之后 `ptr_to_int(p)` | ❌ | ⚠️ 需追加 P3 | P3：把 `ptr_to_int` 下沉到 `scf.if`/`scf.for` 结果（复用 `getIfResultCarrierType` 的 i64 承载思路，扩展到 tensor） |
| **M1** 地址值张量直接访存 | `tl.load(x.to(ptr))`（gather-of-addresses） | ❌ | ❌ | 需方案 B |
| **M2** 地址值张量 + addptr 再访存 | block table / 指针追逐 | ❌ | ❌ | 需方案 B |
| **F1** 指针张量比较 | `p1 == p2` | ❌ | ❌ | 前端 `IncompatibleTypeErrorImpl('invalid operands of type pointer<int32> ...')` 直接拒绝，属 Triton 语义层限制，与后端无关 |

因此对“**用户自己写别的 int_to_ptr 用例会不会出错**”的回答是分层的：

1. **标量 int_to_ptr**：现在就不会出错（S 类，已有专门用例覆盖）。
2. **指针值张量只做数据流**：现在会出错（D1–D4），方案 A 落地后不会出错；D4 需顺带补 P3。
3. **地址值张量真正当访存地址**（M1/M2）：会出错，而且**不是方案 A 的能力范围**，任何局部 patch 都覆盖不了——这是后端“指针=memref 描述符”模型的本质缺口，只能走方案 B。

也就是说：A 的边界由“是否需要真实访存”划开，而不是由“是不是 test_where”划开。M1/M2 在上游 GPU Triton 是合法用法（`tl.load` 加载地址表再解引用），在 Ascend 上确实仍不支持，这一点应在文档/issue 中显式声明，避免用户误以为“int_to_ptr 已全面支持”。

#### A.7 建议的补充改动（让 A 更完整）

- **P3（推荐同批做）**：`ptr_to_int` 下沉 `scf.if`/`scf.for` 结果，覆盖 D4；
- **P4（诊断友好化）**：在 `TritonToLinalg` 前加一次校验，对“A 之后残留的张量 `int_to_ptr`”给出明确报错，例如
  `tensor-typed int_to_ptr used as a memory address is not supported on Ascend (only data-flow / round-trip usage is)`，
  替代当前的 `failed to legalize operation 'tt.int_to_ptr' that was explicitly marked illegal`（后者对用户无信息量，本次排查成本主要就耗在这条错误上）。

### 方案 B（中长期）：把“地址值张量”作为一等公民支持

- 目标：允许 `tensor<Nx!tt.ptr<T>>` 在**只做数据流**（select/cmp/ptr_to_int/跨控制流）时用 `tensor<Nxi64>` 承载地址，而不是 memref。
- 难点：同一 MLIR 类型 `tensor<Nx!tt.ptr<T>>` 在现有转换器里被无条件映射为 `memref<NxT>`（见 §3.2），两种语义共用一个类型，必须先用标记 pass 区分“地址描述（address carrier）”与“地址值（address value）”（仓库里已有 `MarkTensorKind` 之类标记机制可以借鉴/扩展）。
- 改动面：`MemoryTypeConverter`/`TritonTypeConverter` 的类型映射、`Load/Store/Atomic/Gather/Scatter` 等全部 memop converter、`BlockPtrAnalysis`、`PtrToIntConverter`、以及各处“指针必须来自 memref”的断言。
- 收益：可支持“地址表查表后直接访存”等真实场景（需配合 `tt.ascend.indirect_load` 之类逐 lane 地址 helper，或 GM 基址 + 字节偏移的建模）。
- 成本/风险：高，建议作为独立特性立项，不用于本次修复。

### 方案 C（用户侧/前端规避，立即可用）

在该用例的指针分支上避免“指针往返”，直接用整数做 select：

```python
a = tl.load(a_ptr + offsets, mask=mask)            # i64 地址
b = tl.load(b_ptr + offsets, mask=mask)
output = tl.where(decide, a, b)                     # 纯整数 select
tl.store(output_ptr + offsets, output, mask=mask)
```

实测等价形态（能力矩阵 case 1）通过；`test_where[1-int64]` 本身也是通过的。
但这是绕过而非修复，且 `test_core.py` 属于上游测试，不建议改测试来“修 bug”。

### 方案 D（不推荐）：用例标记 xfail / skip

只能掩盖问题，用户写同样代码仍会失败；仅在确实需要“先让 CI 变绿”时临时使用，并需跟踪 issue。

---

## 6. 修复后如何验证

1. 单测回归：
   ```bash
   python -m pytest <lang-triton>/python/test/unit/language/test_core.py -k "test_where" -v
   ```
   期望 13 个组合中 `*int32` 转为通过（`float64` 仍失败，属 BiShengHIR `hivm.hir.load` 不支持 f64，另一个问题）。
2. 探针脚本回归：`probe_pointer_casts.py` 中 case 3/4 应转为 `[OK]`；case 2/5/6 必须保持 `[OK]`。
3. 指针相关既有用例回归（确认没有把“指针值”语义改坏）：
   - `test_inttoptr_offset_reset.py`（标量 int_to_ptr + 广播间接 load）
   - `test_flip.py` / `test_gather.py` / `test_scatter.py` 等含指针张量 select、间接访存的用例
   - 重点观察：`BlockPtrAnalysis.cpp#L706` 的防御性报错是否被触发（说明有路径被改写后仍留下指针张量 select）。
4. IR 抽查：编译 `test_where[1-*int32]` 时 dump TTIR，确认 `tt.int_to_ptr`/`tt.ptr_to_int` 已消失，只剩 `arith.select` + `tt.load`/`tt.store`。

---

## 7. 影响面与风险

| 项 | 评估 |
|----|------|
| 改动范围 | 2 个新 pattern + 1 处注册，均在 `TritonToStructured` 内 |
| 触发条件 | 仅当 `ptr_to_int` 的源是 `int_to_ptr` 或“指针张量 select”时触发 |
| 语义风险 | 低（地址往返恒等、elementwise 交换律） |
| 对现有通过用例的影响 | 理论上为零；风险集中在“指针张量 select 恰好是真实访存基址”的组合，需靠 §6.3 回归确认 |
| 遗留缺口 | M1/M2（地址值张量直接当访存地址）与 D4（跨 `scf.if`，需 P3）仍不支持，见 §A.6 |

---

## 8. 附录

同目录下材料：

- `probe_pointer_casts.py`：6 种指针形态能力探针（含注释说明每个 case 的语义）
- `probe_inttoptr_coverage.py`：张量 `int_to_ptr` 的 8 类消费者探针（C1–C8），支撑 §A.6 覆盖矩阵
- `probe_inttoptr_dataflow.py`：纯数据流变体探针（D3b 多层 select / D4 跨 `scf.if`）
- `dump_c4.py`：指针张量比较（F1）的前端报错取证
- `dump_case3_4.py`：单独复现 case 3 / case 4 的最小脚本
- `case3_4_full.log`：完整报错 + MLIR reproducer（含实际 pass pipeline 与失败时的 TTIR）
- `case3_4_error.log`：异常信息摘要

关键代码位置索引：

- `IntToPtrConverter`：`TritonToLinalg/TritonOpConverter.cpp:3479`
- `PtrToIntConverter`：`TritonToLinalg/TritonOpConverter.cpp:3501`
- `PointerSelectConverter` / `isScalarPointerSelect`：`TritonToLinalg/TritonOpConverter.cpp:433` / `:233`
- 指针类型 → memref 映射：`TritonToLinalg/TritonToLinalgPass.cpp:602-630`
- 动态合法性（pointer-free 判定）：`TritonToLinalg/TritonToLinalgPass.cpp:1086-1092`
- 张量指针 select 防御性报错：`TritonToLinalg/BlockPtrAnalysis.cpp:706`
- 规范化 pattern 注册点：`TritonToStructured/TritonToStructuredPass.cpp:84-104`
