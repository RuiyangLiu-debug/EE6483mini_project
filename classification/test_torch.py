import torch
import sys

print("=" * 60)
print("🔍 PyTorch 环境检测脚本")
print("=" * 60)

# 检查 PyTorch 是否可导入
try:
    print(f"✅ PyTorch 版本: {torch.__version__}")
except Exception as e:
    print("❌ 无法导入 PyTorch！")
    print(e)
    sys.exit(1)

# 检查 CUDA 可用性
cuda_available = torch.cuda.is_available()
print(f"⚙️  CUDA 是否可用: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"🧠 检测到 GPU 数量: {device_count}")
    for i in range(device_count):
        print(f"   ├── GPU {i}: {torch.cuda.get_device_name(i)}")
    print("🚀 正在测试 GPU 计算...")

    # 测试一次张量计算是否真的在 GPU 上执行
    x = torch.rand(3, 3).to("cuda")
    y = torch.rand(3, 3).to("cuda")
    z = torch.mm(x, y)
    print("✅ GPU 矩阵乘法成功！结果张量位于:", z.device)
else:
    print("⚠️ 未检测到可用的 CUDA GPU，正在测试 CPU 计算...")
    x = torch.rand(3, 3)
    y = torch.rand(3, 3)
    z = torch.mm(x, y)
    print("✅ CPU 计算成功。结果张量位于:", z.device)

print("=" * 60)
print("🎯 检测完成！")
