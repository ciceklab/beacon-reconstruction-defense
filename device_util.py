import os
import torch

def get_torch_device(cuda_index: int | None = None) -> torch.device:
    forced = os.environ.get('BEACON_DEVICE', '').strip().lower()
    if forced == 'cpu':
        return torch.device('cpu')
    if forced == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    if forced == 'cuda' or torch.cuda.is_available():
        if not torch.cuda.is_available():
            raise RuntimeError("BEACON_DEVICE=cuda requested but CUDA is not available.")
        idx = cuda_index
        if idx is None:
            env_idx = os.environ.get('BEACON_CUDA_DEVICE', '').strip()
            idx = int(env_idx) if env_idx.isdigit() else 0
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid CUDA device index {idx}; available devices: 0–{torch.cuda.device_count() - 1}."
            )
        torch.cuda.set_device(idx)
        return torch.device(f'cuda:{idx}')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def configure_cpu_threads() -> None:
    n = os.cpu_count() or 4
    os.environ.setdefault('OMP_NUM_THREADS', str(n))
    os.environ.setdefault('MKL_NUM_THREADS', str(n))
    torch.set_num_threads(n)

def announce_device(device: torch.device) -> None:
    if device.type == 'cuda':
        idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.empty_cache()
        print(f'Device set to : cuda:{idx} ({torch.cuda.get_device_name(idx)})')
    elif device.type == 'mps':
        print('Device set to : Apple Metal (MPS)')
    else:
        print('Device set to : cpu')
