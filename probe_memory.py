import torch
from torchvision.models import efficientnet_v2_s

model = efficientnet_v2_s(weights= None).cuda()

batch_size = 16

for size in [224, 288, 320, 384]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        x= torch.randn(batch_size, 3, size, size, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out=model(x)
        out.sum().backward()
        peak=torch.cuda.max_memory_allocated() / 1024**3
        print(f"{size}px batch {batch_size} peak {peak:.2f} GB")

    except torch.cuda.OutOfMemoryError:
        print(f"{size}px batch {batch_size} OOM")