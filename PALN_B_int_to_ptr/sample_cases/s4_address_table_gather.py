"""S4（NVIDIA 场景 N1 / 落点 D）：地址表间接寻址（gather of addresses）。

这是 tensor 形态 int_to_ptr 在 NVIDIA 上的主力场景：整数张量里存的是
逐 lane 的字节地址，转成指针后直接访存。等价于 CUDA 的
``dst[i] = *(T*)addr[i]``。
Ascend：❌ 底层只支持 "单 base + 偏移"，需要 NPUIR 新增 "每 lane 绝对地址
访存" op（见 plan_b.md §12.4 R1）。

期望结果：NVIDIA ✅ ；Ascend（方案 B）❌ 明确报错
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def s4(tab_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    addrs = tl.load(tab_ptr + offs)  # i64 地址表
    v = tl.load(addrs.to(PT))  # 逐 lane 解引用
    tl.store(out_ptr + offs, v)


def main():
    n = 64
    x = torch.arange(n, dtype=torch.int32, device=DEVICE) * 3
    out = empty(n, torch.int32)
    if not launch("S4 地址表 gather", s4, (1,), addr_table(x), out, BLOCK=n):
        return False
    check("S4 地址表 gather", out, x)
    return True


def test_s4():
    main()


if __name__ == "__main__":
    main()
