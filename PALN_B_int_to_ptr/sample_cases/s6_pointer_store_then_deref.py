"""S6（NVIDIA 场景 N1 的真实来源 / 落点 A+D）：内核 A 写指针表，内核 B 读表访存。

这是地址表在生产代码里的典型来源：前一个内核把 ``base + byte_offsets`` 写成
int64 表（A 段，纯数据流），后一个内核把表里的整数转成指针再解引用（D 段）。
Ascend：A 段可过（标量 ptr_to_int + 整数运算），D 段 ❌ 需要 NPUIR。

期望结果：NVIDIA ✅ ；Ascend（方案 B）❌ 明确报错（阻塞在 use_table）
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def build_table(x_ptr, tab_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    base = x_ptr.to(tl.int64)  # 标量 ptr_to_int（现有已支持）
    tl.store(tab_ptr + offs, base + offs * 4)  # int32：4 字节步长


@triton.jit
def use_table(tab_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(tl.load(tab_ptr + offs).to(PT))
    tl.store(out_ptr + offs, v)


def main():
    n = 64
    x = torch.arange(n, dtype=torch.int32, device=DEVICE) * 5 + 1
    tab = empty(n, torch.int64)
    out = empty(n, torch.int32)
    ok = launch("S6a 构造指针表", build_table, (1,), x, tab, BLOCK=n)
    ok = launch("S6b 读表并解引用", use_table, (1,), tab, out, BLOCK=n) and ok
    if not ok:
        return False
    check("S6 跨内核指针表", out, x)
    return True


def test_s6():
    main()


if __name__ == "__main__":
    main()
