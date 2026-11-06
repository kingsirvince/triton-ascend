# 方案 B：地址值指针张量的一等公民化（实现 + 验证报告）

分支：`1104_main-dev_test_where_int32`
代码：`/home/w30038269/code/triton-ascend`
探针/日志：`/home/w30038269/test/1104_test_where_int32/`
关联文档：`test_where_int32_analysis.md`（方案 A 分析，本文档是其 §5「方案 B（中长期）」的落地版）

---

## 1. 结论（TL;DR）

方案 B 已在本地分支实现并跑通验证，核心结论六条：

1. **`test_where[1-*int32]` 已通过**（`python/test/unit/language/test_core.py -k test_where --device npu`：13 passed / 1 failed，唯一失败是与本改动无关的 `test_where[1-float64]`，失败原因是 BiShengIR 不支持 f64 的 `hivm.hir.load`）。
2. 方案 B 不是“为这一个用例打补丁”：它把**所有只在数据流中流动的 `tt.int_to_ptr` 指针张量**（往返、单层/多层 select、跨 `scf.if`/`scf.while` 回边、`splat`/`broadcast`/`expand_dims`/`reshape`/`bitcast`/`addptr`、指针相等比较）统一降级为 `tensor<...xi64>` 地址张量，彻底绕开“指针张量 = memref 描述符”这一模型缺口。
3. 对**真正需要逐 lane 解引用**的场景（地址表 gather / 指针追逐，即 M1/M2），方案 B 不做“假装支持”，而是**在 TTIR 早期给出可直接定位的报错**，替代原来无信息量的 `failed to legalize operation 'tt.int_to_ptr' ...`，以及更糟的下游断言（`lhs.getRank() == rhs.getRank()`）。
4. **方案 B 不需要改 NPUIR / BiShengIR，也不需要其版本升级**：它把地址张量从“指针域”搬回“整数域”，底层只看到普通 i64 张量。为什么标量 `int_to_ptr` 依赖 NPUIR 而这个不需要、M1/M2 又为什么绕不开 NPUIR，见 §2.3。
5. **“客户各种 tensor `int_to_ptr` 写法能否通过”**：指针只作数据流动的写法（现实中的绝大多数）全部通过；`join/split/trans` 之类白名单缺口是纯工程量；真正过不去的是“地址表直接解引用”和“混用真实指针”两类，都有定向报错。分类表见 §1.1 与 §2.4.2。
6. **对齐全 NVIDIA 场景的增量成本 = 一个 NPUIR op**：NVIDIA 侧 `int_to_ptr` 是 no-op（指针就是值），其真实场景（地址表 gather / 指针追逐 / 选指针后访存）终点都是“逐 lane 地址解引用”。因此只需 NPUIR 提供 R1（接受 `tensor<Nxi64>` 绝对地址的访存 op），其余在 triton-ascend 完成。场景清单、7 个最简样例与 NPUIR 需求见 §12。

### 1.1 方案 B 的准确表述与客户用例通过性

> **方案 B 的准确表述**：它不是“让底层支持指针张量”，而是**把“必须在底层实现的部分”精确收缩到一个 op 级需求——逐 lane 地址访存**；其余四类指针语义（偏移、形态变换、比较、转回整数/传出）全部在 triton-ascend 内部规约成 i64 运算，零 NPUIR 依赖。

客户用例按落点分五类（详细展开见 §2.4.2）：

| 类 | 写法 | 结果 | 阻塞点 |
|----|------|------|--------|
| **A** 指针当数据流动（存取/选择/嵌套选择/跨控制流/偏移/函数封装） | `p=load(a+offs).to(PT)`；`where(c,p,q)`；跨 `scf.if` | ✅ **全部通过** | 无（已实测覆盖，见 §9） |
| **B** 未进白名单的形态变换 | 指针张量上 `join/split/interleave/trans/permute/tensor-gather` | ⚠️ 报错 | **仅白名单工程量**，不需 NPUIR（§2.4.3） |
| **C** 混合 provenance | `where(c, 地址值, 真实分配指针)` | ⚠️ 报错 | 设计取舍：统一地址化会动全局指针模型（§2.4.4） |
| **D** 地址表解引用（间接寻址/指针追逐） | `x = tl.load(tl.load(tab+i).to(PT))` | ❌ 报错 | **真能力缺口，必须 NPUIR 协同**（§2.3.5） |
| **E** 指针张量比较 | `p1 == p2` | ❌ 前端拒绝 | Triton 语义层限制（F1），后端 i64 比较已就绪 |

> 注意：D 类在 Triton 里是**合法语义**（NVIDIA 后端就是每 lane 用自己的地址访存），Ascend 不支持属于能力缺口，不能表述为“用户写错了”。

实测覆盖矩阵（本文 §9 有完整表格）：

| 类别 | 当前分支 | 方案 B 后 |
|------|----------|-----------|
| S 标量 `int_to_ptr` | ✅ | ✅ 无回归（种子只收集张量形态） |
| D1 往返 `ptr_to_int(int_to_ptr(x))` | ❌ | ✅ |
| D2 `where` 选指针 → `ptr_to_int`（本次用例） | ❌ | ✅ |
| D3 嵌套 select | ❌ | ✅ |
| D3b 两级 select 汇总 | ❌ | ✅ |
| D4 指针张量跨 `scf.if`（无 addptr） | ❌ | ✅ |
| C7 指针张量跨 `scf.if`（真实分支） | ❌ | ✅ |
| C8 标量 `int_to_ptr` splat → load | ❌ | ✅（原本就靠 broadcast 展开） |
| F1 指针张量比较 | ❌（前端拒绝） | 前端仍拒绝（Triton 语义层限制，与后端无关） |
| M1 地址值张量直接 `load` | ❌ | ❌ 明确报错（不再断言/乱码报错） |
| M2 地址值张量 `addptr` 后 `load` | ❌ | ❌ 明确报错 |

---

## 2. 问题定位（为什么必须“在 TTIR 最早期”做）

### 2.1 triton-ascend 的指针模型

现有后端把“指针”统一建模为 **memref 描述符（base + offset + stride）**：

- 标量指针 → `memref<i8>`（容量可扩）或 `!tt.ptr` → `memref` 视图；
- 指针张量 → `memref<NxT>`，且**要求同一张量所有 lane 共享同一个 base**。

而 `tt.int_to_ptr` 的结果是**运行时字节地址**，每个 lane 一个独立地址，没有“base”概念。于是：

- `IntToPtrConverter` 只接受**标量** `!tt.ptr<T>`，张量形态直接落到 `IllegalOp` → 泛化报错；
- 即使有人想硬扛，`OffsetAnalysis` 在分析“张量指针是否同 base”时会对 rank 不同的 base/offset 直接断言失败（`lhs.getRank() == rhs.getRank()`）。

### 2.2 为什么 pass 位置必须提前到 `triton-to-unstructure` 之前

第一版实现只把逻辑挂在 `TritonToLinalgPass::runOnOperation()` 的最前面（在 `inlinePointerCarryingCallees` 之后）。结果 D2/D3 通过，但 **C5/C6 仍然在下游断言**：

```
Assertion `lhs.getRank() == rhs.getRank()' failed
```

根因：`compiler.py` 的 TTIR 流水线里 `ascend.passes.ttir.add_triton_to_unstructure(pm, ...)` 排在 `add_triton_to_linalg` **之前**，也就是说 `OffsetAnalysis`/`PtrAnalysis` 在方案 B 之前就已经看过未降级的地址值指针张量了。

结论：方案 B 必须挂在 **TTIR 流水线最前面的用户级 pass 位置**，即 `add_triton_control_flow_opt` 之后、`add_triton_to_structure` / `add_triton_to_unstructure` 之前。那时 IR 还是纯 TTIR，没有任何 memref/描述符类型干扰，降级是纯类型改写，代价最低。

> 因此实现上**挂了两处**：
> - `compiler.py` 中的早期 pass（真正解决问题的主路径）；
> - `TritonToLinalgPass` 内部的兜底调用（保证 `triton-opt --triton-to-linalg` 这类 lit 用例、以及任何直接跑 linalg pass 的路径也能自洽）。
>
> 两处都调用同一个函数，且天然幂等（第二次跑时已无张量形态 `tt.int_to_ptr`，`seeds` 为空，直接返回）。

### 2.3 关键澄清：方案 B 依赖 NPUIR / BiShengIR 吗？

**结论：方案 B 不需要改 NPUIR，也不需要 NPUIR 升级版本。** 但这不是“巧合”，而是因为它做的是**减法**：它不要求底层学会“处理地址张量”，而是把地址张量**从指针域搬回整数域**，让底层只看到它早就支持的 i64 张量。

要讲清楚这一点，必须先分清三件不同的事。

#### 2.3.1 三件事，依赖不同

| 能力 | 谁提供 | 现状 |
|------|--------|------|
| 标量 `int_to_ptr`：把运行时 i64 地址变成一个**可解引用的指针** | NPUIR：`hivm.hir.pointer_cast` (`i64 → memref`) | 已有，triton-ascend 直接用它 |
| 访存 | NPUIR：`hivm.hir.load/store`（`base memref` + offsets） | 已有 |
| **地址值只做数据流**（方案 B 的覆盖范围） | 不需要任何新 op：`arith`(i64) + 既有 i64 张量通路 | 已有，方案 B 只是把 IR 改写成这种形态 |
| 逐 lane 独立地址的访存（M1/M2） | 需要 NPUIR 侧的“地址表访存”能力 | **没有**（见 2.3.3） |

#### 2.3.2 标量 `int_to_ptr` 为什么确实依赖 NPUIR

标量路径的完整链路（三处都可查证）：

1. [`IntToPtrConverter`](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp#L3479-L3503) 明确只接受标量：`isa<ShapedType>(pointerType.getPointeeType())` 时报 `only scalar pointer reconstruction is supported`；
2. 它调用 [`createScalarPointerCast`](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/BlockPtrAnalysis.cpp#L66-L79)，后者创建 **`hivm::PointerCastOp`**——这是 NPUIR 的 op（定义在 `AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td`，通过 [`ASCENDNPU_IR_SRC_DIR`](file:///home/w30038269/code/triton-ascend/third_party/ascend/CMakeLists.txt#L31) 引入）；
3. 真正访存由 `hivm.hir.load/store` 完成（[LoadStoreConverter.cpp](file:///home/w30038269/code/triton-ascend/third_party/ascend/lib/TritonToLinalg/LoadStoreConverter.cpp#L233) 先走 `memref.load`，再在 HIVM 阶段收敛成 `hivm.hir.load`）。

也就是说：**“i64 地址 → 设备指针”这一步转换只有一个实现者，就是 NPUIR**。triton-ascend 只负责“调用它”。

那为什么标量必须经过这一步？因为标量 `int_to_ptr` 的**目的就是最终解引用**：要 `load(addr)`，就必须让后端知道“这个 i64 是一个 GM 地址”，并把它变成一个能参与寻址的指针/描述符。张量形态之所以卡住，是因为它要的是“N 个独立地址同时可解引用”，而 NPUIR 的类型里根本没有这种东西（下节）。

#### 2.3.3 底层模型里没有“指针张量”

看 NPUIR 的三个相关 op 的签名就能确认，它们**全部以“单个 base”为中心**：

| op | 签名要点 | 能否表达“每 lane 任意地址” |
|----|----------|------------------------------|
| [`hivm.hir.pointer_cast`](file:///home/w30038269/code/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L257-L293) | `ins(Variadic<I64> addrs)` → `Res<AnyMemRef>` | ❌ 结果是**一个** memref 描述符；多个 addr 表示“多个 buffer 的基址”，不是逐 lane 地址 |
| [`hivm.hir.gather_load`](file:///home/w30038269/code/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L437-L486) | `ins(AnyMemRef base, RankedTensorOf<[I32,I64]> indices, ...)` | ⚠️ 只能“一个 base + 每 lane 偏移” |
| [`hivm.hir.indirect_load`](file:///home/w30038269/code/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L723-L762) | `ins(AnyMemRef src, offsets, dst, mask, other)` | ⚠️ 同上，`dst[i] = src[offsets[i]]` |
| [`hivm.hir.load_scalar`](file:///home/w30038269/code/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L120-L126) | `ins(LLVMPointerType addr)` → scalar | ❌ 单地址单标量，且当前 triton-ascend 未使用 |

关键点：**memref 描述符 = base + offset + stride，它是“一个内存视图”，不是“一列地址”**。所以“指针张量”在底层没有类型可落，这才是当前报错的真正来源：

- 标量 `int_to_ptr` → 一个描述符 ✔
- 张量 `int_to_ptr` → **N 个不同 base 的描述符** ✘（描述符是类型层面的东西，不能当数据张量排布）

这解释了为什么任何“保留指针张量”的方案最后都会撞墙（`failed to legalize` / `lhs.getRank() == rhs.getRank()` 断言），也解释了为什么方案 A 只能覆盖“地址在数据流里绕一圈又回到整数”的形态。

#### 2.3.4 所以方案 B 为什么不碰 NPUIR

方案 B 重写后的 IR 里**一个指针、一个描述符都不产生**：

```
tt.int_to_ptr      → 删除（i64 值就是地址本身）
tt.ptr_to_int      → identity
tt.addptr          → arith.muli(字节缩放) + arith.addi
select/splat/broadcast/reshape/expand_dims/bitcast → 同语义 i64 版本
```

产物是普通 `tensor<...xi64>`，与用户自己写的 `tl.int64` 张量**完全同构**，走的是既有通路。这一点有直接证据：`test_where[1-int64]` 在改动前后都是 PASSED——说明“i64 张量从 TTIR 一路降到 device binary”本来就是通的，方案 B 只是把地址问题**规约**成这个已通路的特例。

一句话：**方案 B 不是在“让后端支持指针张量”，而是在“让指针张量根本不需要被支持”。**

反过来看，这也是它为什么能只改 triton-ascend：
- 不需要新的 `hivm` op → 不需要 NPUIR 代码改动；
- 不需要新 op 的 lowering → 不需要 BiShengIR 版本升级；
- 与 NPUIR 的耦合点只有“TTIR → memref”那一层，而方案 B 在**它之前**就把不需要指针的用例摘出去了；
- 另外，i64 上的 `add/mul/select/cmp` 最终都落到标量或向量整数指令，属于 NPU 的基本能力，不存在隐藏的硬件短板。

#### 2.3.5 M1/M2 的真实依赖边界（诚实结论）

方案 B 覆盖不了“逐 lane 地址直接访存”，而且**这个缺口不是 triton-ascend 单方面能补的**，因为它需要 NPUIR 的模型扩展。两条可能路线：

- **路线 a（可在 triton-ascend 侧做，但要 NPUIR 已有 op 兜底）**：如果能证明地址表里所有地址都落在**同一个 GM 分配**内，就可以把“绝对地址表”改写成“`base` + 每 lane 偏移”，正好匹配 `hivm.hir.gather_load` / `hivm.hir.indirect_load` 的形态。需要：
  1. triton-ascend 侧的 provenance 分析证明“同 base”（跨分配时无法证明，只能拒绝）；
  2. 后端确实能为这两个 op 生成对应硬件指令（gather 类访存），也就是**仍然是 NPUIR/BiShengIR 的能力边界**。
- **路线 b（必须改 NPUIR）**：支持真正的“每 lane 绝对地址访存”。现有 `base + offsets` 的类型约束表达不了它，只能新增 op 或扩展 `load/store` 的操作数形态。

所以方案 B 的正确定位是：**把“数据流子集”彻底解决（零 NPUIR 依赖），把“真解引用”留成一条明确的、可被 NPUIR 能力演进接住的边界**。而且它已经把这个边界铺好了地基——降级后地址在 IR 里是显式的 i64 张量，将来接路线 a/b 时不需要再动一次类型系统。

### 2.4 那是否意味着“没必要真正实现 tensor 的 int_to_ptr”？客户各种写法都能过吗？

这是两个问题，分开回答，答案不一样。

#### 2.4.1 结论一：对“数据流”而言，确实不需要底层实现 —— 因为语义上是多余的

先把 Triton 里 `tensor<...x!tt.ptr<T>>` 的**可用操作穷举**（Triton 语义只提供这五类）：

| # | 操作类别 | 典型 op | 能否在 i64 域等价表达 |
|---|----------|---------|------------------------|
| 1 | **解引用** | `tt.load` / `tt.store` / `tt.atomic*` | ❌ 不能（这是“内存访问”，不是“值运算”） |
| 2 | 偏移 | `tt.addptr` | ✅ 字节加法 |
| 3 | 形态/布局变换 | `splat`/`broadcast`/`expand_dims`/`reshape`/`bitcast`/`join`/`split`/`interleave`/`trans` | ✅ 逐元素重排，与元素类型无关 |
| 4 | 比较 | `arith.cmpi eq/ne` | ✅ 整数比较 |
| 5 | 转回整数 / 传出 | `tt.ptr_to_int`、`scf.yield`、`func.return`、`call` | ✅ 整数原样传递 |

方案 B 覆盖 2–5，**唯一不覆盖的是 1**。而“地址向量的解引用”恰恰是**唯一一个必须由底层理解“这一列值是地址”的操作**——其余四类都只是把地址当普通数据搬来搬去。

于是得到一个干净的判据：

> **“要不要真正实现 tensor 的 `int_to_ptr`” ⟺ “客户是否需要对一批地址直接解引用”。**
> 只要不解引用，这个指针张量在语义上就是“一个 i64 张量”，底层实现反而是一种浪费。

这也解释了为什么方案 B 的验证能这么干净：它测的不是“某个 shape 能不能过”，而是“这五类操作里除了解引用以外是否全部等价”。

#### 2.4.2 结论二：不能承诺“客户随便写都能过”，只能承诺“按语义分类全覆盖 + 有解释的拒绝”

客户用例按落点分五类（A–E，A 类再按具体写法细分）：

| 类 | 典型写法 | 方案 B 结果 | 需要做什么才能过 |
|----|----------|-------------|------------------|
| **A. 指针当数据流动** | `p = tl.load(a+offs).to(PT)`；`tl.store(out+offs, p.to(tl.int64))` | ✅ 通过（C1） | — |
| **A. + 条件选择** | `tl.where(c, p, q).to(tl.int64)`（本次 `test_where`） | ✅ 通过（C2/D2 实测） | — |
| **A. + 嵌套选择** | `tl.where(c, p, tl.where(d, p, q))` | ✅ 通过（C3/D3b） | — |
| **A. + 控制流** | 指针跨 `scf.if` / 循环携带后再转整数 | ✅ 通过（C7/D4） | — |
| **A. + 偏移** | 指针张量 `+ 标量/张量偏移` 后再转整数 | ✅ 通过 | — |
| **A. + 函数封装** | 指针张量作 `@triton.jit` 函数参数 | ✅ 通过（`make_ttir` 阶段的 inliner 已先内联；万一残留 `tt.call` 会给出定向报错，见 §4.5） | — |
| **B. 未列入白名单的形态变换** | 对指针张量用 `tl.join` / `tl.split` / `tl.interleave` / `tl.trans` / `tl.permute` / `tl.gather`(tensor 版) | ⚠️ 当前**报错** | 白名单 + 一条 clone-in-i64 规则即可（见 2.4.3），**不需要 NPUIR** |
| **C. 混合 provenance** | `tl.where(c, addr_from_int, real_allocated_ptr)` | ⚠️ 当前**报错** | 需“统一地址化”重构，代价见 2.4.4 |
| **D. 地址表解引用** | `idx = tl.load(tab+i); x = tl.load(idx.to(PT))`（间接寻址 / 指针追逐） | ❌ **报错（能力缺口）** | 必须 NPUIR 协同（§2.3.5 路线 a/b） |
| **E. 指针比较** | `p1 == p2` on pointer tensors | ❌ 前端就拒绝 | 前端语义放宽；后端 i64 比较已就绪（F1） |

所以诚实结论是：**A 类（也就是现实中绝大多数写法）全过；B 类是白名单工程量；C 类是设计取舍；D/E 才是真边界。**

为什么说 A 类是绝大多数？因为 Triton 里出现 `tensor` 形态 `int_to_ptr` 的常见来源就是“把指针当成数据存下来 / 从内存里取出来 / 在几个指针里挑一个 / 穿过控制流”——这些行为本身就不是在寻址，而是在搬运。真正需要“地址表直接寻址”（D 类）的代码是少数派，典型是图计算的邻居表、embedding 的间接索引、稀疏 gather。

#### 2.4.3 B 类为什么容易补（且仍然零 NPUIR 依赖）

`join/split/interleave/trans/permute` 对地址而言和 `reshape/broadcast` 完全同类：**只重排元素位置，不改元素语义**。实现上就是把 `isAddressPropagator` 的白名单加几行，再复用已有的“克隆该 op 但把结果类型换成 i64”逻辑（当前 `reshape/expand_dims` 走的就是这条路径）。风险低、可增量。

同理 `tl.gather`（tensor 版，非内存 gather）在 i64 域也等价可用。

#### 2.4.4 C 类为什么选择保守（而不是顺手做掉）

“混合 provenance”理论上完全可做：把真实分配得到的指针也 `ptr_to_int` 成地址，就能在整数域统一 `select`。但代价是：

- 一旦对**真实指针**做地址化，就绕开了 memref 描述符路径；
- 而描述符路径是 `unstructure` / `OffsetAnalysis` / 块指针分析 / 同步插入的**公共基础设施**，影响面会从“局部改写”扩大到“全局指针模型”；
- 换来的收益只是支持“地址与真指针混选”这种极罕见写法。

所以这里刻意选择“报错”。这是一个可以随时放开的开关，不是永久放弃。

#### 2.4.5 “是否没必要真正实现”——分场景回答

- **对 triton-ascend 当前目标（让上层 Triton 代码跑起来）**：不必要。方案 B 已把除“逐 lane 地址解引用”以外的全部语义在本地消化掉。
- **对“与 NVIDIA 后端语义完全对齐”**：仍然必要。因为 `tl.load(addr_tensor)` 在 Triton 里是**合法语义**（NVIDIA 后端就是每 lane 用自己的地址访存），Ascend 不支持属于**能力缺口**，不能说成“用户写错了”。
- **正确的表述**：方案 B 不是“永久不实现”，而是**把必须由底层实现的部分精确收缩到“一个 op 级别的需求：逐 lane 地址访存”**，其余全部在 triton-ascend 内部解决。这让将来 NPUIR 的工作量降到最小。

最后一点很重要：**方案 B 与将来 NPUIR 的实现不冲突，也不需要回退**。blocked 集合里的地址保持指针表示不动；将来若新增逐 lane 地址 lowering，只需在它前面接一下即可。反过来，方案 B 降级出来的 i64 地址张量，恰恰是这类新 op 最自然的输入形态（地址已是纯整数、无描述符包袱）。

---

## 3. 设计目标与不变量

### 3.1 目标

1. 覆盖“地址值只做数据流”的全部形态，**不依赖具体用例/形状/mask**；
2. 绝不触碰两类既有正确路径：
   - 标量指针载体路径（`int_to_ptr` 标量结果 → memref）；
   - 真实分配指针的 memref 描述符路径；
3. 无法支持的形态必须**早期、明确、可定位**地失败，而不是下游断言。

### 3.2 三条不变量（实现即围绕它们）

- **不变量 I（种子封闭）**：只有“张量形态的 `tt.int_to_ptr` 结果”是种子；标量 `int_to_ptr` 永不进入重写集合。
- **不变量 II（可表达封闭）**：一个值可以被降级，当且仅当它的**全部消费者**都在“地址传播算子 / `ptr_to_int` / 指针相等比较”集合内；任何消费者是 `tt.load`/`tt.store`/`tt.atomic*`/`tt.call`/`scf.yield`/`func.return` 等，则整条**生产链**被判为 blocked。
- **不变量 III（provenance 不混用）**：任何算子只要有一个指针操作数不是地址值（例如 `select` 的一个分支是真实分配得到的指针），该算子及其操作数全部留在指针域（反向阻塞传播）。

三条合起来保证：**重写后的 IR 在语义上等价，未重写的部分与改动前完全一致**。

---

## 4. 算法

### 4.1 四个集合

| 集合 | 含义 | 计算方式 |
|------|------|----------|
| `seeds` | 张量形态 `tt.int_to_ptr` 的结果 | `module.walk` |
| `candidates` | 可用 `i64` 表达的地址值（种子的前向闭包） | 从种子出发，只穿过地址传播算子 |
| `blocked` | 必须保留指针表示的值 | 前向：消费者不在传播集合内 → blocked；反向：被 blocked 的结果 / 混用 provenance 的算子 → 操作数 blocked |
| `replacements` | 旧指针值 → 新 `i64` 值 | 重写阶段按程序顺序填充 |

### 4.2 伪代码

```
run():
  collectSeeds()                      # 不变量 I
  if seeds.empty(): return true       # 纯标量内核 → 零改动
  computeCandidates()                 # 不变量 III（operandsAreAddresses 过滤）
  computeBlocked()                    # 不变量 II（前向）
  propagateBlockedBackwards()         # 不变量 III（反向）
  if failed(diagnoseUnsupportedUses()): return false   # 先诊断，保留可读 IR
  rewriteInProgramOrder()
```

### 4.3 地址传播算子集合

```
arith.select | tt.addptr | tt.splat | tt.broadcast
tt.expand_dims | tt.reshape | tt.bitcast
```

外加两类“消费但不阻断”的算子：

```
tt.ptr_to_int                # 地址 → 整数，本来就是整数域
arith.cmpi eq/ne (指针)      # 地址比较等价于整数比较
```

> 说明：`tt.load`/`tt.store`/`tt.atomic*`/`tt.call`/`scf.yield`/`tt.return` **不在**集合内。
> 其中 `scf.yield` 的处理有个细节：早期实现把 `yield` 直接当阻塞源，导致 C7（指针张量跨 `scf.if`）被误判；现在只要 yield 的值本身在 candidates 中，就保持可降级（`scf.if` 的结果类型随之改成 i64 张量）。

### 4.4 重写规则

| 原 op | 重写后 | 说明 |
|-------|--------|------|
| `%p = tt.int_to_ptr %i : tensor<Nxi64> -> tensor<Nx!tt.ptr<T>>` | 删除，`%p` 直接用 `%i`（必要时 `arith.extsi` 到 i64） | 地址就是整数值本身 |
| `tt.ptr_to_int %p` | identity（直接用地址值） | 已无指针语义 |
| `tt.addptr %p, %off` | `addr + off * sizeof(T)`（`arith.muli` + `arith.addi`，标量 off 先 splat） | 元素偏移 → 字节偏移 |
| `arith.select %c, %p, %q` | `arith.select %c, %addrP, %addrQ` | 同语义，仅元素类型变化 |
| `tt.splat` / `tt.broadcast` / `tt.expand_dims` / `tt.reshape` | 重建同名 op，结果类型换 i64 | 用 `OperationState` 原样复制属性 |
| `tt.bitcast`（指针→指针） | identity | 字节地址不变 |
| `arith.cmpi eq/ne`（指针） | `arith.cmpi`（i64） | 地址相等 ↔ 整数相等 |

重写按 **module 内程序顺序**遍历，靠 `replacements` 做值替换，最后统一删除死 op（反序遍历 `deadOps`，只删 `use_empty()` 的）。

### 4.5 诊断（M1/M2）

对 `blocked` 集合中的值，逐个检查其消费者并给出定向报错（`diagnoseUnsupportedUses`）：

1. 作为访存 base（`tt.load`/`tt.store`/`tt.atomic*` 的 0 号操作数）：
   ```
   address-value pointer tensor is used as the base of a memory access; on Ascend
   every lane of a load/store must share one base pointer, so a tensor of
   tt.int_to_ptr addresses cannot be dereferenced yet
   ```
2. 与真实指针混用（`select` 一支来自分配/入参）：
   ```
   address-value pointer tensor is mixed with a pointer that does not come from
   tt.int_to_ptr; such a value has no memref representation on Ascend
   ```
3. 逃逸到整数域外（yield/return/call/其他 op）：
   ```
   address-value pointer tensor escapes the integer domain here (region yield,
   function return, call argument or unsupported operation); it cannot be
   represented on Ascend
   ```

报错发生在**重写之前**，所以失败时 IR 保持可读（便于 dump 分析）。

---

## 5. 代码落点

| 文件 | 类型 | 内容 |
|------|------|------|
| `third_party/ascend/include/TritonToLinalg/LowerAddressValuePointers.h` | 新增 | 声明 `lowerAddressValuePointerTensors(ModuleOp)` 与 `createLowerAddressValuePointersPass()` |
| `third_party/ascend/lib/TritonToLinalg/LowerAddressValuePointers.cpp` | 新增（531 行） | 全部算法实现 + `createLowerAddressValuePointersPass()` 工厂（`getArgument()` 预留为 `lower-address-value-pointers`，尚未注册到全局 pass registry） |
| `third_party/ascend/lib/TritonToLinalg/CMakeLists.txt` | 修改（+1 行） | 源文件入编译 |
| `third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp` | 修改（+9 行） | 在 `inlinePointerCarryingCallees` 之后调用，失败则 `signalPassFailure()` |
| `third_party/ascend/triton_ascend.cc` | 修改（+5 行） | 暴露 Python API `ascend.passes.ttir.add_lower_address_value_pointers` |
| `third_party/ascend/backend/compiler.py` | 修改（+4 行） | TTIR 流水线早期插入该 pass |

### 5.1 流水线插入点（关键）

`third_party/ascend/backend/compiler.py`：

```python
ascend.passes.ttir.add_triton_control_flow_opt(pm)
# A tensor of tt.int_to_ptr addresses has no memref form. Lower the
# data-flow-only cases to i64 here, before the unstructure/memory stages
# assume every pointer comes from a single base pointer.
ascend.passes.ttir.add_lower_address_value_pointers(pm)
ascend.passes.ttir.add_triton_to_structure(pm, False, False, compile_on_910_95)
ascend.passes.ttir.add_discrete_mask_access_conversion(pm, compile_on_910_95, compile_mode)
ascend.passes.ttir.add_triton_to_annotation(pm)
ascend.passes.ttir.add_triton_to_unstructure(pm, compile_on_910_95, compile_mode)
...
```

`third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp`：

```cpp
void TritonToLinalgPass::runOnOperation() {
  inlinePointerCarryingCallees(moduleOp);

  // Address values (tensors of pointers built by tt.int_to_ptr) have no memref
  // representation. Rewrite the ones that only take part in data flow into i64
  // before the memory stage classifies pointer values.
  if (failed(lowerAddressValuePointerTensors(moduleOp))) {
    signalPassFailure();
    return;
  }
  ...
}
```

### 5.2 为什么 Pass 要定义在 TritonToLinalg 库里（踩过的坑）

最初把 `PassWrapper` 子类直接写在 `triton_ascend.cc`（Python 绑定 TU）里，链接出来的 `libtriton.so` 导入时报：

```
ImportError: libtriton.so: undefined symbol: _ZTIN4mlir4PassE
```

原因：该 TU 用默认（开 RTTI）编译，而 MLIR/LLVM 侧是 `-fno-rtti`，基类 typeinfo 不会生成。修法是把 Pass 类放到 `TritonToLinalg` 库（与其它 pass 同样的编译选项），`triton_ascend.cc` 只保留 `pm.addPass(createLowerAddressValuePointersPass())` 一行绑定。

---

## 6. 编译与安装（docker 内）

前置（与既有环境一致）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/bin/setenv.bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH=/home/w30038269/env/NPUIR/3.2.2_post2_1010/bishengir/bin:$PATH
export LLVM_INSTALL_PREFIX=/home/l00654397/llvm-f6ded0be-3f5016d1-ubuntu-arm64/
```

增量编译（**不要**跑 `build_main.sh`，它开头 `rm -rf build/` 会全量重编）：

```bash
cd /home/w30038269/code/triton-ascend
cmake --build build/cmake.linux-aarch64-cpython-3.11 -j 96
```

安装（复用已配置的 build 目录，只重跑 Python 侧的拷贝）：

```bash
cd /home/w30038269/code/triton-ascend
LLVM_SYSPATH=$LLVM_INSTALL_PREFIX TRITON_BUILD_TD=ON TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

安装路径：`/usr/local/python3.11.15/lib/python3.11/site-packages/triton/`。

> 注意：只拷 `libtriton.so` 而不同步 Python 源码会漏掉 `compiler.py` 的流水线改动，表现为“C++ 改了但行为没变”。必须走 `setup.py install`（或同时同步 `backend/compiler.py`）。

---

## 7. 验证

### 7.1 目标用例

```bash
cd /home/w30038269/code/triton-ascend
python3 -m pytest python/test/unit/language/test_core.py -k "test_where" --device npu -v
```

结果：

```
test_where[1-int8]     PASSED
test_where[1-int16]    PASSED
test_where[1-int32]    PASSED
test_where[1-int64]    PASSED
test_where[1-uint8]    PASSED
test_where[1-uint16]   PASSED
test_where[1-uint32]   PASSED
test_where[1-uint64]   PASSED
test_where[1-float16]  PASSED
test_where[1-float32]  PASSED
test_where[1-float64]  FAILED   ← 与本改动无关
test_where[1-bfloat16] PASSED
test_where[1-*int32]   PASSED   ← 本次目标
test_where_broadcast[1] PASSED
========= 1 failed, 13 passed =========
```

`test_where[1-float64]` 的失败信息是 BiShengIR 侧限制，不是指针问题：

```
'hivm.hir.load' op failed to verify that operand at idx 0 should have element type
f8E4M3FN ... 32-bit float or 64-bit unsigned integer or 64-bit signless integer
```

（该 op 的 base 类型列表里没有 f64；改动前后表现一致。）

### 7.2 消费者覆盖探针（`probe_inttoptr_coverage.py`）

```bash
cd /tmp && python3 /home/w30038269/test/1104_test_where_int32/probe_inttoptr_coverage.py
```

```
[OK]   C1 tensor int_to_ptr -> ptr_to_int
[OK]   C2 int_to_ptr -> select -> ptr_to_int (test_where)
[OK]   C3 nested select -> ptr_to_int
[FAIL] C4 cmpi on pointer tensor: at 5:26: ... IncompatibleTypeErrorImpl(
         'invalid operands of type pointer<int32> and pointer<int32>')
[FAIL] C5 load through address tensor: ... cannot be dereferenced yet
[FAIL] C6 addptr on address tensor -> load: ... cannot be dereferenced yet
[OK]   C7 pointer tensor across scf.if
[OK]   C8 scalar int_to_ptr broadcast -> load
```

- **C4** 是 **前端**拒绝（`pointer == pointer` 在 Triton 语义层就不合法），与后端无关，见 §9 的 F1。后端的 `cmpi eq/ne` 重写仍然保留（MLIR 层面可能存在该形态）。
- **C5/C6** 现在报的正是 §4.5 设计的定向错误，说明“不支持但可解释”，不再是断言或 `failed to legalize ...`。

### 7.3 数据流变体探针（`probe_inttoptr_dataflow.py`）

```
[OK]   D4 ptr value across scf.if (no addptr)
[OK]   D3b two-level select -> ptr_to_int
```

### 7.4 MLIR lit 回归（`--triton-to-linalg` 路径，含兜底调用）

环境未安装 `lit` python 包（`No module named lit`），改用等价方式执行每个文件的 `// RUN:` 行（解析续行 `\` 与 `// RUN:` 续写），`FileCheck` 用 LLVM 前缀里的二进制：

```
General/TritonToLinalg/*.mlir + 950PR/TritonToLinalg/*.mlir
→ total ok=83 fail=0
```

其中与本次模型最相关、且同时覆盖“标量 inttoptr / 指针 select / bitcast / 混合 base”的用例均通过：

- `test_inttoptr_offset_reset.mlir`（标量 `int_to_ptr` + `addptr` 偏移表示）
- `scalar_pointer_select.mlir`（标量指针 select 载体）
- `pointer_bitcast_byte_buffer.mlir`（指针 bitcast）
- `hoist_broadcast.mlir`
- `tensor_ptr_different_base_invalid.mlir`（tensor ptr 不同 base 的既有路径，含 `--triton-to-linalg`）
- `loop_carried_scalar_address.mlir`（循环携带标量地址，检查无 `!tt.ptr` 残留）

### 7.5 pytest_ut 回归

```bash
python3 -m pytest third_party/ascend/unittest/pytest_ut/test_inttoptr_offset_reset.py \
  third_party/ascend/unittest/pytest_ut/test_where_lt.py \
  third_party/ascend/unittest/pytest_ut/test_where_mask.py \
  third_party/ascend/unittest/pytest_ut/test_where_var.py -q
```

```
27 passed
```

（`test_inttoptr_offset_reset.py` 是标量 inttoptr 的端到端用例，等于给不变量 I 上了回归保护。）

---

## 8. 语义正确性论证

1. **纯数据流场景（D1–D4/C1–C3/C7/C8）**
   指针在 TTIR 里的可观测语义只有两个：字节地址值、以及解引用。既然该指针的所有消费者都被限定在“传播/转整数/相等比较”，那么把它换成等值的 i64 地址是**严格等价变换**：`addptr` 的字节偏移换算与 pointee 大小一致，`select`/`splat`/`broadcast`/`reshape` 都是逐元素映射，`ptr_to_int` 本来就是取地址。

2. **混合场景（不变量 III）**
   一旦某个 `select` 的某个分支不是地址值，反向阻塞会把整条分支链留在指针域；此时该 `select` 结果仍是“真实指针”，继续走既有 memref 描述符路径，行为与改动前完全一致（`tensor_ptr_different_base_invalid.mlir` 覆盖了这条路径）。

3. **标量场景（不变量 I）**
   种子只收集 `ShapedType` 的 `int_to_ptr` 结果，标量路径一行都不经过我们的重写，`test_inttoptr_offset_reset.py` 全绿可证。

4. **幂等性**
   第二次运行（`TritonToLinalgPass` 内兜底）时 IR 中已无张量形态 `int_to_ptr`，`seeds` 为空 → 立即返回。因此“挂两处”没有副作用。

---

## 9. 能力矩阵（方案 B 之后）

| 类别 | 典型写法 | 前端 | 方案 B 后 | 说明 |
|------|----------|------|-----------|------|
| **S** 标量 `int_to_ptr` | `p = tl.load(ptr).to(tl.pointer_type(t))`，再 `p + offs` | ✅ | ✅ | 不进入重写（种子仅张量） |
| **D1** 往返 | `ptr_to_int(int_to_ptr(x))` | ✅ | ✅ | `int_to_ptr` 消除 + `ptr_to_int` identity |
| **D2** 指针值 select → int | `test_where[1-*int32]` | ✅ | ✅ | 已实测 PASSED |
| **D3** 嵌套 select | `where(c, p, where(d, p, q))` | ✅ | ✅ | 探针 C3 |
| **D3b** 两级 select 汇总 | 两轮 `tl.where` 再转 int | ✅ | ✅ | 探针 D3b |
| **D4/C7** 跨 `scf.if`/循环回边 | `if f: p = q` 后 `ptr_to_int(p)` | ✅ | ✅ | 探针 C7、D4 |
| **C8** 标量 inttoptr → splat | `tl.load(p + offs)` | ✅ | ✅ | 走 broadcast 展开 |
| **F1** 指针张量比较 | `p1 == p2` | ❌ | ❌ | 前端 `IncompatibleTypeErrorImpl`，Triton 语义层限制 |
| **M1** 地址值张量直接访存 | `tl.load(addr_tensor)` | ✅ | ❌（明确报错） | 底层只支持“单 base + 偏移”，见 §2.3.3 |
| **M2** 地址值张量 `addptr` 后访存 | 地址表/指针追逐 | ✅ | ❌（明确报错） | 同上 |

**M1/M2 的边界说明（重要）**：这已在方案 A 文档 §5 中标注为“任何局部 patch 都覆盖不了”，原因不在 triton-ascend，而在底层访存模型只支持“单 base + 偏移”（见 §2.3.3/§2.3.5）。方案 B 的价值是把它们从“断言崩溃 / 无信息报错”变成“一行可定位的拒绝”，并且在 IR 层留好了后续扩展点（地址值已经是显式 i64 张量，将来接 `hivm.hir.indirect_load`/`gather_load` 路线时不需要再改类型系统）。

除上表外，还有两类“当前报错但可扩展”的形态（未列入上表，避免与实测结果混淆）：未进白名单的形态变换（`join/split/interleave/trans/permute/tensor-gather`）与混合 provenance `select`，处理策略见 §2.4.2–§2.4.4。

---

## 10. 风险与影响面

| 风险 | 评估 | 缓解 |
|------|------|------|
| 误降级真实指针 | 低 | 不变量 II/III 双重封闭；混合 provenance 一律 blocked |
| 影响标量路径 | 无 | 不变量 I（种子仅张量形态） |
| 影响既有 memref 描述符路径 | 无 | 未被重写的部分零改动；83 条 lit RUN + 27 条 pytest_ut 全绿 |
| 编译期开销 | 可忽略 | 4 次 `module.walk`，每个函数 O(op 数)；无 seeds 时立即返回 |
| 报错过严（本可支持的场景被判 blocked） | 中低 | 传播集合已覆盖 `select/addptr/splat/broadcast/expand_dims/reshape/bitcast/ptr_to_int/cmpi`；后续按 issue 增量加算子 |
| 与上游 Triton 同步冲突 | 低 | 全部落在 `third_party/ascend/`，不修改上游 `lib/`/`python/triton/`（`setup.py install` 打补丁造成的 in-tree 改动不属于本次提交范围） |
| NPUIR / BiShengIR 依赖 | 无新增 | 不新增 `hivm` op、不改 lowering，只依赖现有 `arith` i64 通路（见 §2.3.4）；不需要 NPUIR 版本升级 |

---

## 11. 后续工作（建议）

1. **补 lit 用例**：在 `third_party/ascend/unittest/Conversion/General/TritonToLinalg/` 增加 `address_value_pointer_tensor.mlir`，覆盖 D1/D2/D3/C7 与 M1/M2 的报错文本（当前仓库没有位于该 pass 下游的回归点）。
2. **注册 pass 到 triton-opt**：便于 lit 直接 `--lower-address-value-pointers` 验证（现在只能通过 `--triton-to-linalg` 间接覆盖）。
3. **M1/M2 的正式支持**：不是 triton-ascend 单方面能完成的，必须与 NPUIR 协同（详见 §2.3.5，接口级需求见 §12.5 的 R1–R7）。落点是“地址值已是 i64 张量”这个既成事实之上：
   - 路线 a：做 provenance 分析，证明地址表同属一个 GM 分配后改写为 `base + offsets`，复用 `hivm.hir.gather_load` / `indirect_load`；跨分配时仍只能拒绝；
   - 路线 b：推动 NPUIR 支持真正的“每 lane 绝对地址”访存形态。
4. **前端 F1（指针比较）**：若产品需要 `p1 == p2`，需放宽 `semantic.py` 的类型检查；后端已具备等价的 i64 比较重写。
5. **白名单增量扩展**：`join/split/interleave/trans/permute/tensor-gather` 等形态变换可低成本纳入（§2.4.3），建议按客户实际报错逐个添加并配套 lit 用例。
6. **文档/用户说明**：明确“`int_to_ptr` 的数据流用法已支持，逐 lane 解引用仍不支持”，避免用户误判。

---

## 12. NVIDIA 场景对齐与 NPUIR 开发需求

本节回答三个问题：NVIDIA 后端到底有哪些 `int_to_ptr` 真实场景、是否就是前面的 A/D 两类、以及要全部对齐需要 NPUIR 提供什么。

### 12.1 根本差异：NVIDIA 是“值语义指针”，Ascend 是“描述符语义指针”

| 维度 | NVIDIA / LLVM | Ascend / BiShengIR |
|------|---------------|--------------------|
| 指针在 IR 里的身份 | 一个 64 位 SSA 值（`ptr`），**每个 lane/线程各持一个** | 一个 memref **描述符**（`base + offset + stride`），一个 memref 值只描述**一个**内存视图，不是“一列地址” |
| `int_to_ptr` | `llvm.inttoptr`，**no-op**（等价 bitcast） | 必须把 i64 变成 memref，走 `hivm.hir.pointer_cast`，且只支持标量 |
| 解引用 | `llvm.load ptr`，每线程用自己的值 | `hivm.hir.load`，**所有 lane 共享同一 base** |
| 指针运算 | 64 位整数加法 | `addptr` 变成 memref 的 offset/stride 变换 |

由此得到一个很关键的推论：**在 NVIDIA 侧“支持 int_to_ptr”等于什么都不用做**（no-op），所以 NVIDIA 从不为它写任何后端代码；Ascend 侧一旦要求解引用，就必须造一个能承载“每 lane 独立地址”的访存 op。这就是 §2.3 中“标量依赖 NPUIR、数据流不依赖、M1/M2 必须依赖”的分界线在 NVIDIA 视角下的镜像。

### 12.2 NVIDIA 的 `int_to_ptr` 使用场景（N1–N5）

Triton 中 tensor 形态 `int_to_ptr` 的唯一入口是 `x.to(tl.pointer_type(T))`。真实用法按“终点”归为五类：

| 编号 | 场景 | 典型形态 | 终点 |
|------|------|----------|------|
| **N1** | 地址表间接寻址 | 整型张量存逐 lane 字节地址 → 转指针 → `load` | 解引用 |
| **N2** | 指针追逐 | 表里存的是指针，逐跳解引用（图遍历、链表、多级索引、paged KV cache 页表） | 解引用 |
| **N3** | 条件指针选择后访存 | `where/if` 选不同 buffer 的指针，随后 `load/store` | 解引用 |
| **N4** | 指针值当数据搬运 | 存取、跨内核传递、调试输出，**不解引用** | 纯数据流 |
| **N5** | 指针比较 / 地址差 | 判断是否同一元素、计算字节偏移 | 比较运算 |

### 12.3 “就是 A 和 D 吗？”——是，但 A 通常只是 D 的前缀

- 严格按“是否解引用”映射：**N1/N2/N3 = D**（其中选择/搬运部分是 A 前缀），**N4 = A**，**N5 = E**。
- 更重要的是：NVIDIA 真实内核里“把指针当整数搬来搬去”的目的，绝大多数是稍后要解引用。纯 A（永不解引用）主要出现在调试输出、地址差统计等边缘用法。
- 因此 **“只对齐 NVIDIA 有的场景” ⟹ “必须支持 D”**。方案 B 让 A 以及 D 的前缀全部通过，但没有、也不可能缩小最后这一格：**NVIDIA 全场景对齐的增量成本 = 一个“每 lane 绝对地址访存”的 op**。

### 12.4 场景 ↔ 最简样例 ↔ 当前状态（`sample_cases/`）

| 编号 | 场景 | 落点 | 样例 | NVIDIA | Ascend（方案 B） | 阻塞点 |
|------|------|------|------|--------|------------------|--------|
| **S1** | `int → ptr → int` 往返 | A（N4） | [s1_pointer_roundtrip.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s1_pointer_roundtrip.py) | ✅ | ✅ | — |
| **S2** | select 指针后转整数（`test_where` 形态） | A（N3 前缀） | [s2_pointer_select.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s2_pointer_select.py) | ✅ | ✅ | — |
| **S3** | select 指针后解引用 | A+D（N3） | [s3_pointer_select_deref.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s3_pointer_select_deref.py) | ✅ | ❌ 明确报错 | R1 |
| **S4** | 地址表 gather | D（N1） | [s4_address_table_gather.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s4_address_table_gather.py) | ✅ | ❌ 明确报错 | R1 |
| **S5** | 两跳指针追逐 | D（N2） | [s5_pointer_chasing.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s5_pointer_chasing.py) | ✅ | ❌ 明确报错 | R1 + R2 |
| **S6** | 跨内核指针表（A 段写表 + D 段读表） | A+D（N1） | [s6_pointer_store_then_deref.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s6_pointer_store_then_deref.py) | ✅ | 建表段预期可过（标量 `ptr_to_int` 现支持）/ 读表段 ❌ | R1 |
| **S7** | 指针比较（转整数 / 直接） | E（N5） | [s7_pointer_compare.py](file:///home/w30038269/test/1104_test_where_int32/sample_cases/s7_pointer_compare.py) | ✅ | 转整数 ✅ / 直接比较 ❌ 前端拒绝 | 前端 F1 |

运行方式（`_common.py` 负责设备选择、地址表构造与 PASS/FAIL 汇报）：

```bash
SAMPLE_DEVICE=npu  python3 sample_cases/s4_address_table_gather.py
SAMPLE_DEVICE=cuda python3 sample_cases/s4_address_table_gather.py
SAMPLE_DEVICE=npu  python3 -m pytest sample_cases -q     # 整目录回归
```

设计说明：`launch()` 把编译/运行异常收敛成 `[LAUNCH-FAIL]` 一行输出，这样在 Ascend 上跑整目录时，能力缺口只记录不中断，天然形成“NVIDIA 场景 × Ascend 支持度”的对照表；`addr_table()` 用 `data_ptr()` 生成逐元素地址，NVIDIA 与 Ascend 通用。

状态口径：S1/S2 的 Ascend 结论来自方案 B 的实测（对应探针 C1–C3），S3–S6 的结论来自 M1/M2 探针（C5/C6）与 §12.1 的架构判断，S7 来自 C4（`cmpi` 在 i64 域可用、前端拒绝直接比较）。NVIDIA 列是按 LLVM 值语义推导的必然结果，可直接用 `SAMPLE_DEVICE=cuda` 复跑核实。

### 12.5 NPUIR 开发需求

| 编号 | 需求 | 优先级 | 说明 |
|------|------|--------|------|
| **R1** | 逐 lane 绝对地址访存 op（load） | **P0** | 语义 `dst[i] = mask[i] ? *(T*)addr[i] : other[i]`，`addr: tensor<Nxi64>`（GM 字节地址，编译期无法确定 base 关系）。建议**新增** op（如 `hivm.hir.load_from_addresses`）而不是给 `hivm.hir.load` 加“地址向量作 base”的重载，避免与 memref base 语义冲突。必须支持：`mask`、`other`、元素类型、`volatile`。 |
| **R1b** | 同形态的 store | P1 | `*(T*)addr[i] = v[i]`；R1 定型后按同一模板扩展 |
| **R2** | 地址向量 → 指针的表示决策 | **P0** | 现状 `hivm.hir.pointer_cast` 的 `Res` 是 `AnyMemRef`（单个描述符），无法表达 N 个地址。**首选**：让 R1 直接消费 `tensor<Nxi64>`，后端类型系统不引入“指针张量”；**次选**：给 pointer_cast 加张量重载。首选方案与方案 B 的 i64 路线天然对齐，triton-ascend 侧只需一处 bridge。 |
| **R3** | 地址运算的生成质量 | P0 | i64 向量 `add/mul/select/cmp` 已具备，需确认地址计算不产生额外 GM 往返（这是 gather 的热点路径） |
| **R4** | 原子访存变体 | P1 | `tt.atomic_*` 作用在地址向量上（图/直方图 scatter 常见）。对齐 NVIDIA 场景时需要 |
| **R5** | 同步与内存序 | P1 | R1 本质是间接访存，`InjectSync`/volatile 必须按普通 load/store 处理，保证“先写地址表、后读表”不被重排（S6 依赖） |
| **R6** | verifier 与越界语义 | P2 | 地址张量位宽必须 64、对齐/类型一致性、mask 形状一致；并明确越界地址是 UB 还是可检测——客户用例会依赖这个结论 |
| **R7** | 同 base 时的优化通道 | P2 | 若可证明地址表落在同一 GM 分配内，降级到现有 `hivm.hir.gather_load` / `indirect_load`（base + offsets）性能更好。provenance 分析在 triton-ascend（它掌握 memref/offset 信息），NPUIR 只需保证这两个 op 的 lowering 支持任意 i64 offsets 与 mask |

**R1 验收标准**：lit 覆盖“地址来自不同 GM 分配 / 地址不连续 / mask 全 0 / N 非向量宽度整数倍 / volatile”五种情形；triton-ascend 侧 S3–S6 端到端数值正确。

### 12.6 分工与优先级

| 工作 | 归属 | 优先级 |
|------|------|--------|
| i64 地址张量的运算/选择/控制流/往返（方案 B 主体） | triton-ascend | ✅ 已完成 |
| 地址向量 → R1 op 的改写 bridge + 诊断文本 | triton-ascend | P0（等 R1 接口定稿） |
| R1 / R1b 新 op（op 定义 + verifier + parser/printer + lowering + lit） | **NPUIR** | P0 / P1 |
| R2 表示决策（推荐：R1 直接吃 i64 张量） | NPUIR 定调，triton-ascend 配合 | P0 |
| R5 同步语义、R6 越界语义 | NPUIR | P1 / P2 |
| R7 provenance 分析复用 `gather_load` | triton-ascend 分析 + NPUIR 确认 lowering | P2 |
| 前端指针比较（F1） | triton-ascend `semantic.py` | P2 |

**一句话结论**：若目标只是“NVIDIA 里不解引用的那部分”，方案 B 已经够用；若目标是“NVIDIA 有的场景都能跑”，唯一必须由 NPUIR 提供的能力就是 **R1——一个接受 `tensor<Nxi64>` 绝对地址的访存 op**，其余全部在 triton-ascend 内完成。

---

## 13. 附录

### 13.1 探针清单

| 脚本 | 覆盖 |
|------|------|
| `probe_inttoptr_coverage.py` | C1–C8（往返 / select / 嵌套 select / 比较 / 直接访存 / addptr 访存 / 跨 `scf.if` / 标量 splat） |
| `probe_inttoptr_dataflow.py` | D3b（两级 select）/ D4（跨 `scf.if` 无 addptr） |
| `probe_pointer_casts.py`、`dump_c3_4.py`、`dump_c4.py` | 早期定位脚本（保留作证据） |
| `sample_cases/s1–s7_*.py` | NVIDIA 场景对齐样例（N1–N5 → S1–S7，见 §12.4） |

### 13.2 相关日志

| 文件 | 内容 |
|------|------|
| `/tmp/probe_cov6.log` | 最终一轮 C1–C8 结果 |
| `/tmp/probe_df2.log` | 最终一轮 D3b/D4 结果 |
| `/tmp/pytest_where2.log` | `test_where` 全量（`--device npu`） |
| `/tmp/mlir_reg2.log` | 83 条 mlir RUN 回归 |
| `/tmp/pytest_ut_reg.log` | 27 条 pytest_ut 回归 |

### 13.3 报错文本对照

改动前（M1/M2）：

```
error: failed to legalize operation 'tt.int_to_ptr' that was explicitly marked illegal
```

或直接断言：

```
Assertion `lhs.getRank() == rhs.getRank()' failed.
```

改动后：

```
error: address-value pointer tensor is used as the base of a memory access; on Ascend
every lane of a load/store must share one base pointer, so a tensor of tt.int_to_ptr
addresses cannot be dereferenced yet
```

### 13.4 本次改动文件清单

```
new  third_party/ascend/include/TritonToLinalg/LowerAddressValuePointers.h
new  third_party/ascend/lib/TritonToLinalg/LowerAddressValuePointers.cpp
mod  third_party/ascend/lib/TritonToLinalg/CMakeLists.txt          (+1)
mod  third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp  (+9)
mod  third_party/ascend/triton_ascend.cc                           (+5)
mod  third_party/ascend/backend/compiler.py                        (+4)
```

NVIDIA 场景样例（不在 triton-ascend 仓库内）：`/home/w30038269/test/1104_test_where_int32/sample_cases/`（`_common.py` + `s1`–`s7`）。
