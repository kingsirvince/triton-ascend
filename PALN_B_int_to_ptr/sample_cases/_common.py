"""sample_cases 公共工具：设备选择、地址表构造、结果校验。

用法::

    SAMPLE_DEVICE=npu python3 s4_address_table_gather.py
    SAMPLE_DEVICE=cuda python3 s4_address_table_gather.py
    SAMPLE_DEVICE=npu python3 -m pytest sample_cases -q      # 整目录回归

默认自动选择：能 import torch_npu 且可用则用 npu，否则用 cuda。
"""

import os

import torch


def _pick_device():
    want = os.environ.get("SAMPLE_DEVICE", "").lower()
    if want in ("npu", "cuda"):
        return want
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("no NPU/CUDA device found; set SAMPLE_DEVICE=npu|cuda")


DEVICE = _pick_device()
if DEVICE == "npu":
    import torch_npu  # noqa: E402,F401


def sync():
    if DEVICE == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def addr_scalar(t):
    """标量地址（单元素 int64 张量），供内核 tl.load(...) 后 .to(pointer)。"""
    return torch.tensor([t.data_ptr()], dtype=torch.int64, device=DEVICE)


def addr_table(t, n=None):
    """逐元素地址表：第 i 个元素是 t[i] 的字节地址（t 需 1D 连续）。"""
    n = t.numel() if n is None else n
    base = t.data_ptr()
    step = t.element_size()
    return torch.tensor(
        [base + i * step for i in range(n)], dtype=torch.int64, device=DEVICE
    )


def empty(n, dtype):
    return torch.empty(n, dtype=dtype, device=DEVICE)


def check(name, out, ref):
    ok = torch.equal(out, ref)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    if not ok:
        print(f"  got={out[:8].tolist()} ref={ref[:8].tolist()}", flush=True)
    assert ok, name


def launch(name, kernel, grid, *args, **kwargs):
    """启动内核；异常转成可读信息（Ascend 上用于记录能力缺口而不是崩溃）。"""
    try:
        kernel[grid](*args, **kwargs)
        sync()
        print(f"[LAUNCH-OK]   {name}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        text = "\n".join(x.strip() for x in str(exc).splitlines())
        print(f"[LAUNCH-FAIL] {name}: {text[:400]}", flush=True)
        return False
