"""S5（NVIDIA 场景 N2 / 落点 D）：两跳指针追逐（指针表里存的是指针）。

图遍历 / 链表 / 多级索引的抽象形态：第一跳读地址，第二跳再读地址，第三跳取值。
NVIDIA：每 lane 独立地址访存，天然支持。
Ascend：❌ 需要 NPUIR 的逐 lane 地址访存（且要求跨 base 也能工作，见 §12.4 R1/R2）。

期望结果：NVIDIA ✅ ；Ascend（方案 B）❌ 明确报错
"""

import torch
import triton
import triton.language as tl

from _common import DEVICE, addr_table, check, empty, launch

PT = tl.pointer_type(tl.int32)


@triton.jit
def s5(top_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mid_addrs = tl.load(top_ptr + offs)  # 第 1 跳：table -> mid 的地址
    leaf_addrs = tl.load(mid_addrs.to(PT))  # 第 2 跳：mid 里存 leaf 的地址
    v = tl.load(leaf_addrs.to(PT))  # 第 3 跳：取叶子值
    tl.store(out_ptr + offs, v)


def main():
    n = 64
    leaf = torch.arange(n, dtype=torch.int32, device=DEVICE) + 7
    mid = addr_table(leaf)  # mid[i] = &leaf[i]（i64）
    top = addr_table(mid)  # top[i] = &mid[i]（i64）
    out = empty(n, torch.int32)
    if not launch("S5 两跳指针追逐", s5, (1,), top, out, BLOCK=n):
        return False
    check("S5 两跳指针追逐", out, leaf)
    return True


def test_s5():
    main()


if __name__ == "__main__":
    main()
