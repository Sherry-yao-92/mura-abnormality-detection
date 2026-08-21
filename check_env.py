import torch
import torch.nn as nn

print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("device     :", torch.cuda.get_device_name(0))
print("arch list  :", torch.cuda.get_arch_list())
print("VRAM (GB)  :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print()

# 1. 矩陣乘法（全連接層在做的事）
x = torch.randn(2048, 2048, device="cuda")
r = (x @ x).mean()
torch.cuda.synchronize()
print("[1] matmul          ok :", float(r))

# 2. 卷積 + 反向傳播（訓練在做的事）
conv = nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(torch.randn(8, 3, 224, 224, device="cuda"))
y.sum().backward()
torch.cuda.synchronize()
print("[2] conv + backward ok")

# 3. fp16 混合精度（你 8GB 的救命稻草）
with torch.autocast("cuda", dtype=torch.float16):
    z = conv(torch.randn(8, 3, 224, 224, device="cuda"))
torch.cuda.synchronize()
print("[3] fp16 AMP        ok , dtype =", z.dtype)