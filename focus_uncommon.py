import torch as th
import torch.utils.data as D
from torchvision.models import resnet50
from torchvision import transforms as T
from experiments.dataset import Focus
from tqdm import tqdm
import sys

ckpt = th.load("Pretrained_models/imagenet_1k_sd.pth", "cpu")
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
    #0: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 6, 7}},
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


# Create a dictionary to map field names to their corresponding lists
field_lists = {
    "time": times,
    "weather": weathers,
    "locations": locations,
}

correct = 0
total = 0

# Open a file for writing
with open('output_log_uncommon_2.txt', 'w') as file:
    # Redirect standard output to the file
    sys.stdout = file

    dataloaders = {}  # Dictionary to store DataLoaders for each category
    correct_dataloader = {}
    total_dataloader = {}
    accuracies_dataloader = {}

    for category, settings in tqdm(uncommon.items()):
        category_name = categories[category]
        #print("category: ", category, "settings: ", settings)

        # Iterate over the specified fields and create a custom Focus for each combination
        # Check which field is specified and set others to None
        specified_field = None
        for field, value in settings.items():
            if value:
                #print("value: ", value)
                specified_field = field
                field_list = field_lists.get(specified_field)
                #print(type(values))
                # Create a DataLoader for each specified field value
                for value in settings[specified_field]:
                    #print(value)
                    dataloader_key = f"{category_name}_{specified_field}_{field_list[value]}"
                    correct_dataloader[dataloader_key] = 0
                    total_dataloader[dataloader_key] = 0

                    custom_focus = Focus(
                        '/data/vilab09/focus/',
                        categories=[category_name],
                        times=None if specified_field != "time" else [times[value]],
                        weathers=None if specified_field != "weather" else [weathers[value]],
                        locations=None if specified_field != "locations" else [locations[value]],
                        transform=test_transform)

                    dataloaders[dataloader_key] = D.DataLoader(
                        custom_focus, batch_size=1, num_workers=1, pin_memory=True
                    )

                    # Iterate over the test DataLoader
                    with th.no_grad():
                        for batch_idx, databatch in enumerate(tqdm(dataloaders[dataloader_key], leave=False)):
                            images, ground_truth_categories, ground_truth_times, ground_truth_weathers, ground_truth_locations = (
                                databatch[0].cpu(), databatch[1].cpu(), databatch[2].cpu(), databatch[3].cpu(), databatch[4].cpu()
                            )

                            # Print information for each iteration
                            print(f"\nCategory: {category_name}")
                            print(f"Dataloader_key: {dataloader_key}")
                            print(f"Batch Index: {batch_idx + 1}/{len(dataloaders[dataloader_key])}")
                            print(f"Images Shape: {images.shape}")
                            print(f"Ground Truth Categories: {categories[ground_truth_categories.item()]}")
                            print(f"Ground Truth Times: {times[ground_truth_times.item()]}")
                            print(f"Ground Truth Weathers: {weathers[ground_truth_weathers.item()]}")
                            print(f"Ground Truth Locations: {[locations[i] for i in range(len(locations)) if ground_truth_locations[0][i] == 1]}")

                            # Forward pass through the network
                            outputs = net(images)

                            # Get the predicted categories
                            _, predicted = th.max(outputs, 1)

                            # Update total count
                            total += images.shape[0]
                            total_dataloader[dataloader_key] += images.shape[0]

                            # Update correct count
                            correct += (predicted == ground_truth_categories).sum().item()
                            correct_dataloader[dataloader_key] += (predicted == ground_truth_categories).sum().item()

                            # Print information for each batch
                            print(f"Predicted Categories: {predicted}")
                            print(f"Accuracy for Batch: {100 * (predicted == ground_truth_categories).sum().item() / images.shape[0]:.2f}%")

                    accuracies_dataloader[dataloader_key] = ""

                    if total_dataloader[dataloader_key] != 0:
                        accuracy_percentage = 100 * correct_dataloader[dataloader_key] / total_dataloader[dataloader_key]
                        accuracies_dataloader[dataloader_key] = f"{accuracy_percentage:.2f}%"
                    else:
                        accuracies_dataloader[dataloader_key] = "N/A (total is zero)"


    print(f"total_dataloader: {total_dataloader}")
    print(f"correct_dataloader: {correct_dataloader}")
    print(f"accuracies_dataloader: {accuracies_dataloader}")
    if total != 0:
        # Print overall accuracy
        print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")
    else:
        print("\nNo batches processed. Overall accuracy cannot be calculated.")

    # Restore the original sys.stdout
    sys.stdout = sys.__stdout__

# Check if total is not zero before calculating overall accuracy
if total != 0:
    # Print overall accuracy
    print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")
else:
    print("\nNo batches processed. Overall accuracy cannot be calculated.")
