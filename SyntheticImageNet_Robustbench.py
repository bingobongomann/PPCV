import numpy as np 
import torch as th
import robustbench as rb
from torchvision.models import resnet50
import torchvision.transforms as transforms

def main():
    #load the model with params from the Paper[2]
    ckpt = th.load("Models/Synthetic/imagenet_1k_sd.pth", "cpu")
    net = resnet50()
    net.fc = th.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
    msg = net.load_state_dict(ckpt, strict=True)
    #check if loading worked
    print(msg)

    # define the preprocessing as defined in the Paper[2]
    Preprocessing = transforms.Compose([
        transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    #set model to eval mode
    net.eval()

    # run robustbench attack with L2 Norm attack model and imagenet data 
    clean_acc, robust_acc = rb.eval.benchmark(net,dataset='imagenet',threat_model="L2", preprocessing=Preprocessing,n_examples=5000)
    print(f"clean acc: {clean_acc} and robust acc: {robust_acc}")

if __name__=="__main__":
    main()