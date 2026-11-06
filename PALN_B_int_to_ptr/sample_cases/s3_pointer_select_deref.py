"""S3（NVIDIA 场景 N3 / 落点 A→D）：条件选择指针后立即解引用。

NVIDIA 上最常见的 "条件访存" 写法（如 where 选 buffer 再 load），
每 lane 用自己的指针值访存，天然支持。
Ascend：A 部分已被方案 B 覆盖，但终点是解引用 -> 需要 NPUIR（见 S4/S5）。

期望结果：NVIDIA ✅ ；Ascend（方案 B）❌ 明确报错
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def s3(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    c = tl.load(c_ptr + offs)
    pa = tl.load(a_ptr + offs).to(PT)
    pb = tl.load(b_ptr + offs).to(PT)
    v = tl.load(tl.where(c != 0, pa, pb))
    tl.store(out_ptr + offs, v)


def main():
    n = 64
    a = torch.arange(n, dtype=torch.int32, device=DEVICE)
    b = torch.arange(n, dtype=torch.int32, device=DEVICE) + 1000
    c = torch.zeros(n, dtype=torch.int8, device=DEVICE)
    c[::2] = 1
    out = empty(n, torch.int32)
    if not launch(
        "S3 select(ptr)->load", s3, (1,), addr_table(a), addr_table(b), c, out, BLOCK=n
    ):
        return False
    check("S3 select(ptr)->load", out, torch.where(c.to(torch.bool), a, b))
    return True


def test_s3():
    main()


if __name__ == "__main__":
    main()
