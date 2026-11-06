"""S1（NVIDIA 场景 N4 / 落点 A）：int -> ptr -> int 往返，纯数据流。

NVIDIA：llvm.inttoptr / llvm.ptrtoint 都是 no-op（指针就是 64 位值），零成本。
Ascend：方案 B 已支持（内核内不产生真正的指针，地址始终是 i64 张量）。

期望结果：NVIDIA ✅ ；Ascend（方案 B）✅
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def s1(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    addrs = tl.load(x_ptr + offs)  # i64 地址
    p = addrs.to(PT)  # tensor 形态 int_to_ptr
    tl.store(out_ptr + offs, p.to(tl.int64))  # ptr_to_int


def main():
    n = 64
    addrs = torch.tensor(
        [0x1000 + i * 4 for i in range(n)], dtype=torch.int64, device=DEVICE
    )
    out = empty(n, torch.int64)
    if not launch("S1 往返 int->ptr->int", s1, (1,), addrs, out, BLOCK=n):
        return False
    check("S1 往返 int->ptr->int", out, addrs)
    return True


def test_s1():
    main()


if __name__ == "__main__":
    main()
