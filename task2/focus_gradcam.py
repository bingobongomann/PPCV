# Adapted from https://github.com/priyathamkat/focus/blob/main/src/experiments/grad_cam_visualizations.ipynb
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights, convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision import transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import numpy as np
import cv2

from experiments.evaluate_model import label_to_correct_idxes
from experiments.imagenet_labels import imagenet_labels
from experiments.dataset import Focus, split_dataset, DCR

num_examples_to_find = 6
model_checkpoint = "Pretrained_models/imagenet_1k_sd.pth"
focus_root = Path("/data/vilab09/focus/")
use_imagenet_labels = True


#ckpt = torch.load(model_checkpoint, "cpu")
#base_model = models.resnet50()
#base_model.fc = torch.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
#msg = base_model.load_state_dict(ckpt, strict=True)

def get_crop_size(model_name):
    size = 224
    if model_name in ["resnet50", "convnext_base", "convnext_large"]:
        size = 232
    elif model_name == "convnext_tiny":
        size = 236
    elif model_name == "convnext_small":
        size = 230

    return size

def init_imagenet_sd(num_classes, checkpoint_path):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path if checkpoint_path is not None else model_checkpoint, "cpu")
    base_model = resnet50()
    base_model.fc = torch.nn.Linear(2048, num_classes, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
    base_model.load_state_dict(ckpt, strict=True)

    return base_model

def init_resnet():
    return resnet50(ResNet50_Weights.IMAGENET1K_V2)

def init_convnext_tiny():
    return convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

def init_convnext_small():
    return convnext_small(ConvNeXt_Small_Weights.IMAGENET1K_V1)

def init_convnext_base():
    return convnext_base(ConvNeXt_Base_Weights.IMAGENET1K_V1)

def init_convnext_large():
    return convnext_large(ConvNeXt_Large_Weights.IMAGENET1K_V1)

def init_model(model_name, num_classes, checkpoint_path, device):

    if model_name.startswith("imagenet"):
        if model_name.endswith("_sd"):
            base_model = init_imagenet_sd(num_classes, checkpoint_path)
    if model_name.startswith("resnet"):
            base_model = init_resnet()
    if model_name.startswith("convnext"):
        if "tiny" in model_name:
            base_model = init_convnext_tiny()
        elif "small" in model_name:
            base_model = init_convnext_small()
        elif "base" in model_name:
            base_model = init_convnext_base()
        elif "large" in model_name:
            base_model = init_convnext_large()
        

    base_model.to(device)
    base_model.eval()

    return base_model

def parse_args():
    parser = argparse.ArgumentParser(description='Script for evaluating a model on a dataset with specified settings.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model. Available: imagenet_100_sd, imagenet_1k_sd, resnet50 (pretrained with imagenet1kv2), convnext_tiny, convnext_small, convnext_base, convnext_large')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='Path to the checkpoint file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify the device (e.g., "cuda:0" or "cpu").')

    return parser.parse_args()

args = parse_args()
device = torch.device(args.device)

num_classes = 1000

# Determine the size of the Linear layer based on the checkpoint file
num_classes = 100 if 'imagenet_100' in args.model_name else 1000

# Initialize ResNet model
base_model = init_model(args.model_name, num_classes, args.checkpoint_path, args.device)

# Define transformations
#test_transform = T.Compose([
#    T.Resize(get_crop_size(args.model_name), interpolation=T.InterpolationMode("bicubic")),
#    T.CenterCrop(224),
#    T.ToTensor(),
#    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#])

#device = "cuda:0"
#base_model.to(device)
#base_model.eval()

gb_base_model = GuidedBackpropReLUModel(model=base_model, use_cuda=False)

base_cam = GradCAM(model=base_model,
              target_layers=[base_model.layer4[-1]],
              use_cuda=False)

test_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



#gb_finetuned_model = GuidedBackpropReLUModel(model=finetuned_model, use_cuda=True)

base_cam = GradCAM(model=base_model,
              target_layers=[base_model.layer4[-1]],
              use_cuda=False)

#finetuned_cam = GradCAM(model=finetuned_model,
#                  target_layer=finetuned_model.layer4[-1],
#                  use_cuda=True)

test_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# Define categories, times, weathers, locations, and settings
categories = ["truck", "car", "plane", "ship", "cat", "dog", "horse", "deer", "frog", "bird"]
times = ["day", "night", "none"]
weathers = ["cloudy", "foggy", "partly cloudy", "raining", "snowing", "sunny", "none"]
locations = ["forest", "grass", "indoors", "rocks", "sand", "street", "snow", "water", "none"]

def is_correct_label(idx, ground_truth):
    #print(f'is_correct_label: idx: {idx} ', f' gt ={ground_truth}')
    return idx in label_to_correct_idxes[ground_truth]


def check_label(top5_prediction, ground_truth, use_strict=False):
    """Returns "is top1 correct?", "is top5 correct?"""
    if use_strict:
        return top5_prediction[0] == ground_truth, ground_truth in top5_prediction
    else:
        return (
            top5_prediction[0].item() in label_to_correct_idxes[ground_truth],
            not set(top5_prediction.tolist()).isdisjoint(
                label_to_correct_idxes[ground_truth]
            ),
        )
    
def overlay_cam_image(cam, input_tensor, target_category, pil_image, custom_target):
    #print("target_category: ", target_category)
    if custom_target is not None:
        print("targets: ", custom_target)
        targets = [ClassifierOutputTarget(custom_target)]
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True)
    elif custom_target == None:
        print("targets: ", target_category)
        targets = [ClassifierOutputTarget(target_category)]
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True)

    
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(np.array(pil_image).astype(np.float32) / 255.0,
                                  grayscale_cam,
                                  use_rgb=True)
    return cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam]), cam_image

def generate_cam_images(bg_var_test_dataset, model, cam, correct, custom_target):
    cam_images = {"correct": [], "incorrect": []}
    gb_cam_images = {"correct": [], "incorrect": []}
    target_categories = {"correct": [], "incorrect": []}
    gt_labels = {"correct": [], "incorrect": []}
    image_paths = {"correct": [], "incorrect": []}
    #for image_idx, (tensor_image, label, _, _, _) in enumerate(bg_var_test_dataset):
    for image_idx, (tensor_image, ground_truth_categorie, _, _, _) in enumerate(bg_var_test_dataset):
        #label = ground_truth_categories
        #tensor_image = images
        try:
            image_file = bg_var_test_dataset.image_files[image_idx][0][1:]
        except AttributeError:
            image_file = bg_var_test_dataset.dataset.image_files[bg_var_test_dataset.indices[image_idx]][0][1:]
        image_path = focus_root / image_file
        print("image_path: ", image_path)
        pil_image = Image.open(image_path)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
            ]
        )(pil_image)
        input_tensor = torch.unsqueeze(tensor_image, 0).to(device)
        outputs = model(input_tensor)
        #_, idxs = torch.max(outputs, dim=1)

        # Get the predicted categories
        _, predicted = torch.topk(outputs, k=5)
        # Build Predicted categories string
        #predicted_labels_names = [imagenet_labels[label] for label in predicted[0].tolist()]

        output_labels = []

        for label in predicted[0].tolist():
            for category, label_set in label_to_correct_idxes.items():
                if label in label_set:
                    output_labels.append(f"{imagenet_labels[label]} ({categories[category]})")
                    break
            else:
                output_labels.append(imagenet_labels[label])

        #print("output_labels: ", output_labels)

        #print("predicted[0]", predicted[0])

        correct_top1_batch, correct_top5_batch = check_label(predicted[0], ground_truth_categorie)

        idx = predicted[0][0].item()
        idx_2 = predicted[0][1].item()

        #print("gen_cam_img: idxs: ", idxs, " label: ", label)
        #print("gen_cam_img: idx: ", idx)
        if not correct_top1_batch and len(cam_images["incorrect"]) != num_examples_to_find:
            image_paths["incorrect"].append(image_path)
            target_categories["incorrect"].append(output_labels[0] if use_imagenet_labels else DCR.classes[idx])
            gt_labels["incorrect"].append(DCR.classes[ground_truth_categorie])

            cam_mask, cam_image = overlay_cam_image(cam, input_tensor, idx, pil_image, custom_target)
            cam_gb = deprocess_image(cam_mask * gb_base_model(input_tensor, target_category=idx))
            cam_images["incorrect"].append(cam_image)
            gb_cam_images["incorrect"].append(cam_gb)
#         elif len(cam_images["incorrect"]) == num_examples_to_find:
#             break
            

        elif correct_top1_batch and len(cam_images["correct"]) != num_examples_to_find:
            image_paths["correct"].append(image_path)
            target_categories["correct"].append(output_labels[0] if use_imagenet_labels else DCR.classes[idx])
            gt_labels["correct"].append(DCR.classes[ground_truth_categorie])

            cam_mask, cam_image = overlay_cam_image(cam, input_tensor, idx, pil_image, custom_target)
            cam_gb = deprocess_image(cam_mask * gb_base_model(input_tensor, target_category=idx))
            cam_images["correct"].append(cam_image)
            gb_cam_images["correct"].append(cam_gb)
        elif len(cam_images["incorrect"]) == num_examples_to_find and len(cam_images["correct"]) == num_examples_to_find:
            break
            
    return cam_images, gb_cam_images, target_categories, gt_labels, image_paths, len(cam_images[correct])

bg_var_test_dataset = Focus(
    focus_root,
    categories=categories,
    times=None,
    weathers=None,
    locations=None,
    transform=test_transform
)
train_set, test_set = split_dataset(bg_var_test_dataset, train_fraction=0.7)
#candidates = find_candidates(test_set)
#print(candidates[650:700])

def evaluate_combination(categories, time, weather, location, correct, custom_target):
    bg_var_test_dataset = Focus(
        focus_root,
        categories=categories,
        times=[time] if time is not None else None,
        weathers=[weather] if weather is not None else None,
        locations=[location] if location is not None else None,
        transform=test_transform
    )
    print("len(bg_var_test_dataset): ", len(bg_var_test_dataset))
    gb_cam_images, cam_images, target_categories, gt_labels, image_paths, num_examples_found = generate_cam_images(bg_var_test_dataset, base_model, base_cam, correct, custom_target)
    print("eval(): num_examples_found: ", num_examples_found)
    for image_path in image_paths[correct]:
        #print(BUCKET_URL / image_path.parents[0].name / image_path.name)
        print(Path(image_path.parents[0].name).joinpath(image_path.name))
    num_cols = 2
    num_rows = num_examples_found
    #print("num_examples_found", num_examples_found)
    das = gb_cam_images[correct][0]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), constrained_layout=True)
    fig.suptitle(f"{correct} classified Images in {time}-{weather}-{location}", fontsize=16)

    # If axs is one-dimensional, reshape it to two-dimensional
    if num_rows == 1:
        axs = axs.reshape(1, -1)

    save_dir = "gradcam_results"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_rows):
        axs[i, 0].grid(False)
        axs[i, 1].grid(False)

        axs[i, 0].imshow(gb_cam_images[correct][i])
        axs[i, 0].title.set_text(f"GT: {gt_labels[correct][i]}, Predicted: {target_categories[correct][i]}"[:35])
        axs[i, 1].imshow(cam_images[correct][i])
        axs[i, 1].title.set_text(f"Guided GradCAM")

    # Save the figures in the "gradcam_results" directory
    save_path = os.path.join(save_dir, f"GC_{args.model_name}_{categories}_{time}_{weather}_{location}_{correct}_{custom_target}.png")
    plt.savefig(save_path)


if __name__ == "__main__":
    evaluate_combination(
    categories=["deer"],
    time="day",
    weather="cloudy",
    location="street",
    correct="correct", #correct or incorrect for prediction
    custom_target=353  #None for setting the highest output as the target or set the targetlabel from 0 to 999
)


