import torch
import torch.nn as nn
import torchvision.models as models
import dcflib as dcf

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

num_classes = 100
m = 9
m1 = 3
kc = 4
nlayer = "coeff_bases_l1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ Replace modules in ConvNeXt ===============================
print("Replacing ConvNeXt-b...")
# 1. Load pretrained ConvNeXt
model = models.convnext_base(pretrained=True).to(device)
for n, p in model.named_parameters():
    p.requires_grad = False
key_list = [key for key, _ in model.named_modules()]
for key in key_list:
    parent, target, target_name = _get_submodules(model, key)
    # print(target_name)
    if isinstance(target, nn.Conv2d):
        new_module = dcf.DCFConv2d(target, m=m, m1=m1, kc=kc, nlayers=nlayer)
        setattr(parent, target_name, new_module)
    if isinstance(target, nn.Linear) and "classifier" not in target_name:
        new_module = dcf.DCFLinear(target, kc=kc)
        setattr(parent, target_name, new_module)

# Replace classifier head
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
cnt = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        cnt += p.numel()
        # print(n, p.shape)
print("Num of params: ", cnt)


# ================ Replace modules in ViT ===============================
print("Replacing ViT-b...")
model = models.vit_b_16(pretrained=True).to(device)
for n, p in model.named_parameters():
    p.requires_grad = False
key_list = [key for key, _ in model.named_modules()]
for key in key_list:
    parent, target, target_name = _get_submodules(model, key)
    # print(target_name)
    if isinstance(target, nn.Conv2d):
        new_module = dcf.DCFConv2d(target, m=m, m1=m1, kc=kc, nlayers=nlayer)
        setattr(parent, target_name, new_module)
    if isinstance(target, nn.Linear) and "classifier" not in target_name:
        new_module = dcf.DCFLinear(target, kc=kc)
        setattr(parent, target_name, new_module)
# Replace classifier head
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
cnt = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        cnt += p.numel()
        # print(n, p.shape)
print("Num of params: ", cnt)



# ================ Replace modules in ViT ===============================
print("Replacing Stable Diffusion...")
m = 9
m1 = 3
kc = 4
nlayer = "bases_l1"
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion v1.5 (can use local path or Hugging Face hub)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # or a local directory path
    torch_dtype=torch.float16,
).to(device)
model = pipe.unet
for n, p in model.named_parameters():
    p.requires_grad = False
key_list = [key for key, _ in model.named_modules()]
for key in key_list:
    parent, target, target_name = _get_submodules(model, key)
    if isinstance(target, nn.Conv2d):
        new_module = dcf.DCFConv2d(target, m=m, m1=m1, kc=kc, nlayers=nlayer)
        setattr(parent, target_name, new_module)
    # if isinstance(target, nn.Linear):
    #     new_module = dcf.DCFLinear(target, kc=kc)
    #     setattr(parent, target_name, new_module)
cnt = 0
for n, p in model.named_parameters():
    if p.requires_grad:
        cnt += p.numel()
        # print(n, p.shape)
print("Num of params: ", cnt)