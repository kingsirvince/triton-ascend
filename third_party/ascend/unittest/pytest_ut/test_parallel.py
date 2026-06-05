# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    SUB_BLOCK_SIZE: tl.constexpr = BLOCK_SIZE // 2
    for s in extension.parallel(0, 2, bind_sub_block=True):
        start = s * SUB_BLOCK_SIZE
        offsets = start + tl.arange(0, SUB_BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, x, mask=mask)

TEST_SIZES = [256, 512, 1024, 2048]

@pytest.mark.parametrize("N", TEST_SIZES)
def test_copy_kernel(N):
    x = torch.arange(N, dtype=torch.float32, device='npu')
    y = torch.empty_like(x)

    copy_kernel[(1,)](x, y, N, BLOCK_SIZE=N)

    assert torch.allclose(x, y), f"Copy failed for N={N}"
    print(f"Test passed for N={N}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])