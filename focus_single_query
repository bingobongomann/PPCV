import torch as th
import torch.utils.data as D
from torchvision.models import resnet50
from torchvision import transforms as T
from experiments.dataset import Focus
from experiments.evaluate_model import check_label
from tqdm import tqdm
import sys

ckpt = th.load("/visinf/home/vilab10/Pretrained_models/imagenet_1k_sd.pth", "cpu")
net = resnet50()
net.fc = th.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
msg = net.load_state_dict(ckpt, strict=True)

net.eval()

test_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
)

focus = Focus(
    "/data/vilab09/focus/",
    categories=[
        "truck"
    ],
    times=["night"],
    weathers=None,
    locations=None,
    #transform=None
    transform=test_transform
)

categories = [
    "truck",
    "car",
    "plane",
    "ship",
    "cat",
    "dog",
    "horse",
    "deer",
    "frog",
    "bird",
]
times = [
        "day",
        "night",
        "none",
]

weathers = [
    "cloudy",
        "foggy",
        "partly cloudy",
        "raining",
        "snowing",
        "sunny",
        "none",
]

locations = [
    "forest",
    "grass",
    "indoors",
        "rocks",
        "sand",
        "street",
        "snow",
        "water",
        "none",
]

uncommon = {
    0: {"time": {1}, "weather": {}, "locations": {}},
    1: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 6, 7}},
    2: {"time": {1}, "weather": {1, 3, 4}, "locations": {0, 2, 3, 4, 6, 7}},
    3: {"time": {1}, "weather": {1, 3, 4}, "locations": {0, 1, 2, 3, 4, 5, 6}},
    4: {"time": {1}, "weather": {1, 3, 4}, "locations": {0, 3, 4, 6, 7}},
    5: {"time": {1}, "weather": {1, 3, 4}, "locations": {0, 3, 6}},
    6: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 5, 6, 7}},
    7: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 4, 5, 6, 7}},
    8: {"time": {}, "weather": {1, 3, 4}, "locations": {2, 5, 6}},
    9: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 5, 6}},
}

test_dataloader = D.DataLoader(
        focus, batch_size=1, num_workers=1, pin_memory=True
    )

correct = 0
correct_top1 = 0
correct_top5 = 0
total = 0

# Open a file for writing
with open('output_log.txt', 'w') as file:
    # Redirect standard output to the file
    sys.stdout = file
    
    # Iterate over the test DataLoader
    with th.no_grad():
        for batch_idx, databatch in enumerate(tqdm(test_dataloader, leave=False)):
            images, ground_truth_categories, ground_truth_times, ground_truth_weathers, ground_truth_locations = (
                databatch[0].cpu(), databatch[1].cpu(), databatch[2].cpu(), databatch[3].cpu(), databatch[4].cpu()
            )
            
            # Print information for each iteration
            print(f"\nBatch Index: {batch_idx + 1}/{len(test_dataloader)}")
            print(f"Images Shape: {images.shape}")
            print(f"Ground Truth Categories: {categories[ground_truth_categories.item()]}")
            print(f"Ground Truth Times: {times[ground_truth_times.item()]}")
            print(f"Ground Truth Weathers: {weathers[ground_truth_weathers.item()]}")
            print(f"Ground Truth Locations: {[locations[i] for i in range(len(locations)) if ground_truth_locations[0][i] == 1]}")

            
            # Forward pass through the network
            outputs = net(images)
            
            # Get the predicted categories
            _, predicted = th.topk(outputs, k=5)
            print("predicted: ", predicted[0])

            # Update total count
            total += images.shape[0]

            # Update correct count
            correct_top1_batch, correct_top5_batch = check_label(predicted[0], ground_truth_categories)

            # Print information for each batch
            print(f"Predicted Categorie: {predicted[0].tolist()}")
            print(f"Top-1 Accuracy for Batch: {100 * correct_top1_batch:.2f}%")
            print(f"Top-5 Accuracy for Batch: {100 * correct_top5_batch:.2f}%")

            # Accumulate correct counts
            correct_top1 += correct_top1_batch
            correct_top5 += correct_top5_batch
    
    # Restore the original sys.stdout
    sys.stdout = sys.__stdout__

# Check if total is not zero before calculating overall accuracy
if total != 0:
    # Print overall accuracy
    print(f"\nOverall Top-1 Accuracy: {100 * correct_top1 / total:.2f}%")
    print(f"\nOverall Top-5 Accuracy: {100 * correct_top5 / total:.2f}%")
else:
    print("\nNo batches processed. Overall accuracy cannot be calculated.")
