import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch_npu

# 检查 NPU 是否可见
print(f"NPU available: {torch.npu.is_available()}")

# 检查设备数量
print(f"Device count: {torch.npu.device_count()}")

# 尝试在 NPU 上创建一个张量 (Hello World!)
x = torch.randn(2, 3).npu()
print(x)
print(x.device)
