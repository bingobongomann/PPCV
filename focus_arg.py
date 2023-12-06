import argparse
import torch as th
import torch.utils.data as D
from torchvision.models import resnet50
from torchvision import transforms as T
from experiments.dataset import Focus
from experiments.evaluate_model import check_label
from tqdm import tqdm
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Script for evaluating a model on a dataset with specified settings.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--settings', type=str, choices=['common', 'uncommon'], default='uncommon', help='Specify whether to use common or uncommon settings.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify the device (e.g., "cuda:0" or "cpu").')
    return parser.parse_args()

args = parse_args()
device = th.device(args.device)

# Load checkpoint
ckpt = th.load(args.checkpoint_path, "cpu")

# Determine the size of the Linear layer based on the checkpoint file
num_classes = 100 if 'imagenet_100_sd' in args.checkpoint_path else 1000

# Initialize ResNet model
net = resnet50()
net.fc = th.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
net.load_state_dict(ckpt, strict=True)
net.to(args.device)
net.eval()

# Define transformations
test_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define categories, times, weathers, locations, and settings
categories = ["truck", "car", "plane", "ship", "cat", "dog", "horse", "deer", "frog", "bird"]
times = ["day", "night", "none"]
weathers = ["cloudy", "foggy", "partly cloudy", "raining", "snowing", "sunny", "none"]
locations = ["forest", "grass", "indoors", "rocks", "sand", "street", "snow", "water", "none"]

uncommon = {
    0: {"time": {1}, "weather": {1, 3, 4}, "locations": {2, 3, 6, 7}},
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

##actually common
common = {
    0: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 4, 5, 8}},
    1: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 4, 5, 8}},
    2: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {1, 5, 8}},
    3: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {7, 8}},
    4: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {1, 2, 5, 8}},
    5: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {1, 2, 4, 5, 7, 8}},
    6: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 4, 8}},
    7: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 8}},
    8: {"time": {0, 1, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 3, 4, 7, 8}},
    9: {"time": {0, 2}, "weather": {0, 2, 5, 6}, "locations": {0, 1, 4, 7, 8}},
}

# Choose the settings based on the command line argument
settings_dict = {'common': common, 'uncommon': uncommon}[args.settings]

# Create a dictionary to map field names to their corresponding lists
field_lists = {
    "time": times,
    "weather": weathers,
    "locations": locations,
}

# Adjust the output filename based on the used checkpoint and setting
checkpoint_name = args.checkpoint_path.split('/')[-1].split('.')[0]
output_filename = f"output_log_{checkpoint_name}_{args.settings}.txt"

# initialise variables
correct_top1 = 0
correct_top5 = 0
total = 0

# Open a file for writing
with open(output_filename, 'w') as file:
    # Redirect standard output to the file
    sys.stdout = file

    dataloaders = {}  # Dictionary to store DataLoaders for each category
    correct_top1_dataloader = {}
    correct_top5_dataloader = {}
    total_dataloader = {}
    accuracies_top1_dataloader = {}
    accuracies_top5_dataloader = {}

    for category, settings in tqdm(settings_dict.items()):
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
                    correct_top1_dataloader[dataloader_key] = 0
                    correct_top5_dataloader[dataloader_key] = 0
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
                                databatch[0].to(args.device), databatch[1].to(args.device), databatch[2].to(args.device),
                                databatch[3].to(args.device), databatch[4].to(args.device)
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
                            _, predicted = th.topk(outputs, k=5)

                            # Update total count
                            total += images.shape[0]
                            total_dataloader[dataloader_key] += images.shape[0]
                            #print("images.shape[0]: ", images.shape[0])

                            # Update correct count
                            print("predicted: ", predicted[0])
                            correct_top1_batch, correct_top5_batch = check_label(predicted[0], ground_truth_categories)

                            correct_top1_dataloader[dataloader_key] += correct_top1_batch
                            correct_top5_dataloader[dataloader_key] += correct_top5_batch


                            # Accumulate correct counts
                            correct_top1 += correct_top1_batch
                            correct_top5 += correct_top5_batch

                            # Print information for each batch
                            print(f"Predicted Categorie: {predicted[0].tolist()}")
                            print(f"Top-1 Accuracy for Batch: {100 * correct_top1_batch:.2f}%")
                            print(f"Top-5 Accuracy for Batch: {100 * correct_top5_batch:.2f}%")


                    accuracies_top1_dataloader[dataloader_key] = ""
                    accuracies_top5_dataloader[dataloader_key] = ""

                    if total_dataloader[dataloader_key] != 0:
                        accuracy_top1_percentage = 100 * correct_top1_dataloader[dataloader_key] / total_dataloader[dataloader_key]
                        accuracies_top1_dataloader[dataloader_key] = f"{accuracy_top1_percentage:.2f}%"
                        accuracy_top5_percentage = 100 * correct_top5_dataloader[dataloader_key] / total_dataloader[dataloader_key]
                        accuracies_top5_dataloader[dataloader_key] = f"{accuracy_top5_percentage:.2f}%"
                    else:
                        accuracies_top1_dataloader[dataloader_key] = "N/A (total is zero)"


    print(f"\ntotal_dataloader: {total_dataloader}")
    print(f"\ncorrect_top1_dataloader: {correct_top1_dataloader}")
    print(f"correct_top5_dataloader: {correct_top5_dataloader}")
    print(f"\naccuracies_top1_dataloader: {accuracies_top1_dataloader}")
    print(f"accuracies_top5_dataloader: {accuracies_top5_dataloader}")
    if total != 0:
        # Print overall accuracy
        print(f"\nOverall Top-1 Accuracy: {100 * correct_top1 / total:.2f}%")
        print(f"Overall Top-5 Accuracy: {100 * correct_top5 / total:.2f}%")
    else:
        print("\nNo batches processed. Overall accuracy cannot be calculated.")

    # Restore the original sys.stdout
    sys.stdout = sys.__stdout__

# Check if total is not zero before calculating overall accuracy
if total != 0:
    # Print overall accuracy
    print(f"\nOverall Top-1 Accuracy: {100 * correct_top1 / total:.2f}%")
    print(f"Overall Top-5 Accuracy: {100 * correct_top5 / total:.2f}%")
else:
    print("\nNo batches processed. Overall accuracy cannot be calculated.")


# Print settings information
print(f"Checkpoint Path: {args.checkpoint_path}")
print(f"Settings: {args.settings}")
