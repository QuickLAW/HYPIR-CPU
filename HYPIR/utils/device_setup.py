"""
Device configuration helpers.

This module standardizes CPU runtime tuning for PyTorch workloads.
It aligns thread pools, enables MKL-DNN, and sets environment hints
to reduce contention and improve deterministic performance on multi-core CPUs.
"""

def setup_cpu_device():
    """Apply CPU-specific runtime configuration.

    - Force device override to CPU for tiled VAE utilities
    - Set PyTorch intra-op threads to logical CPU count; keep inter-op low
    - Enable MKL-DNN backend for optimized ops
    - Configure OMP/MKL/KMP env vars to reduce thread oversubscription
    """
    import torch
    from HYPIR.utils.tiled_vae import devices
    import os
    n = os.cpu_count()
    
    # Set device override for downstream modules
    devices.set_device_override("cpu")
    
    # PyTorch threading and MKL-DNN
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    # Runtime hints for OpenMP/Intel runtime to avoid context switching
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_PROC_BIND", "true")
    
    print("CPU device configuration applied:")
    print(f"  - Device: {devices.device}")
    print(f"  - DType: {devices.dtype}")
    print(f"  - PyTorch threads: {torch.get_num_threads()}")
    print(f"  - MKL-DNN: {torch.backends.mkldnn.enabled}")

def setup_device(device_name):
    """Apply device-specific configuration by name."""
    if device_name == "cpu":
        setup_cpu_device()
    else:
        # GPU configuration remains default; only set override if available
        from HYPIR.utils.tiled_vae import devices
        if hasattr(devices, 'set_device_override'):
            devices.set_device_override(device_name)
        print(f"GPU device configuration applied: {device_name}")
