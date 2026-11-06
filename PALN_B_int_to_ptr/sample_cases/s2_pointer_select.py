"""S2（NVIDIA 场景 N3 / 落点 A）：条件选择指针后仍只做数据流（不访存）。

这就是 test_where 的形态，也是 NVIDIA 上 "select buffer pointer" 的最简骨架。
NVIDIA：select 直接作用在指针值上，llvm.select 即可，零成本。
Ascend：方案 B 已支持（指针已被降级为 i64，select 变成整数 select）。

期望结果：NVIDIA ✅ ；Ascend（方案 B）✅
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def s2(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    c = tl.load(c_ptr + offs)
    pa = tl.load(a_ptr + offs).to(PT)
    pb = tl.load(b_ptr + offs).to(PT)
    p = tl.where(c != 0, pa, pb)
    tl.store(out_ptr + offs, p.to(tl.int64))


def main():
    n = 64
    a = torch.arange(n, dtype=torch.int32, device=DEVICE)
    b = torch.arange(n, dtype=torch.int32, device=DEVICE) + 1000
    ta, tb = addr_table(a), addr_table(b)
    c = torch.zeros(n, dtype=torch.int8, device=DEVICE)
    c[::2] = 1
    out = empty(n, torch.int64)
    if not launch("S2 select(ptr)->int", s2, (1,), ta, tb, c, out, BLOCK=n):
        return False
    ref = torch.where(c.to(torch.bool), ta, tb)
    check("S2 select(ptr)->int", out, ref)
    return True


def test_s2():
    main()


if __name__ == "__main__":
    main()
