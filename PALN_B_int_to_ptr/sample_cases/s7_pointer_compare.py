"""S7（NVIDIA 场景 N5 / 落点 E）：指针比较。

NVIDIA 上 ``pa == pb`` 直接作用在指针值上（icmp ptr），也可先转整数再比较。
Ascend：后端已能生成等价的 i64 比较，但本仓库的 Triton 前端会先报
``IncompatibleTypeErrorImpl``；转整数再比较是可用的替代写法。

期望结果：NVIDIA ✅ ；Ascend：compare_via_int ✅ / compare_direct ❌ 前端拒绝
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def compare_via_int(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pa = tl.load(a_ptr + offs).to(PT)
    pb = tl.load(b_ptr + offs).to(PT)
    tl.store(out_ptr + offs, (pa.to(tl.int64) == pb.to(tl.int64)).to(tl.int8))


@triton.jit
def compare_direct(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    pa = tl.load(a_ptr + offs).to(PT)
    pb = tl.load(b_ptr + offs).to(PT)
    tl.store(out_ptr + offs, (pa == pb).to(tl.int8))


def main():
    n = 64
    a = torch.arange(n, dtype=torch.int32, device=DEVICE)
    b = torch.arange(n, dtype=torch.int32, device=DEVICE)
    ta = addr_table(a)
    tb = addr_table(b)  # 与 b 同址
    out = empty(n, torch.int8)
    ok = launch("S7a 指针转整数后比较", compare_via_int, (1,), ta, tb, out, BLOCK=n)
    if ok:
        check("S7a 指针转整数后比较", out, torch.ones(n, dtype=torch.int8, device=DEVICE))
    else:
        return False
    if launch("S7b 指针直接比较", compare_direct, (1,), ta, tb, out, BLOCK=n):
        check("S7b 指针直接比较", out, torch.ones(n, dtype=torch.int8, device=DEVICE))
    return True


def test_s7():
    main()


if __name__ == "__main__":
    main()
