"""
设备配置补丁 - 在应用启动时调用此函数来正确配置CPU模式
"""

def setup_cpu_device():
    """设置CPU设备配置"""
    import torch
    from HYPIR.utils.tiled_vae import devices
    
    # 设置设备覆盖
    devices.set_device_override("cpu")
    
    # 应用CPU优化设置
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.backends.mkldnn.enabled = True
    
    import os
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    
    print("CPU设备配置已应用:")
    print(f"  - 设备: {devices.device}")
    print(f"  - 数据类型: {devices.dtype}")
    print(f"  - PyTorch线程数: {torch.get_num_threads()}")
    print(f"  - MKL-DNN: {torch.backends.mkldnn.enabled}")

def setup_device(device_name):
    """根据设备名称设置配置"""
    if device_name == "cpu":
        setup_cpu_device()
    else:
        # GPU模式的配置保持不变
        from HYPIR.utils.tiled_vae import devices
        if hasattr(devices, 'set_device_override'):
            devices.set_device_override(device_name)
        print(f"GPU设备配置已应用: {device_name}")
