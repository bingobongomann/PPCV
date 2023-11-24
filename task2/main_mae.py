import torch as th
import torch.utils.data as D
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T
from experiments.dataset import Focus
from tqdm import tqdm

net = resnet50(ResNet50_Weights.IMAGENET1K_V2)
net.fc = th.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"

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
        "truck",
        "car",
        "plane",
        "ship",
        "cat",
        "horse",
        "horse",
        "deer",
        "frog",
        "bird",
    ],
    times=["day"],
    weathers=["sunny"],
    locations=["grass", "street"],
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
    0: {  # truck
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    1: {  # car
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    2: {  # plane
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 2, 3, 4, 6, 7},
    },
    3: {  # ship
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 1, 2, 3, 4, 5, 6},
    },
    4: {  # cat
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 4, 6, 7},
    },
    5: {  # dog
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 6},
    },
    6: {  # horse
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 5, 6, 7},
    },
    7: {  # deer
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 4, 5, 6, 7},
    },
    8: {  # frog
        "time": {},
        "weather": {1, 3, 4},
        "locations": {2, 5, 6},
    },
    9: {  # bird
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 5, 6},
    },
}

test_dataloader = D.DataLoader(
        focus, batch_size=1, num_workers=1, pin_memory=True
    )

correct = 0
total = 0

with th.no_grad():
    for databatch in tqdm(test_dataloader, leave=False):
        images, categories, times, weathers, locations = databatch[0].cpu(), databatch[1].cpu(), databatch[2].cpu(), databatch[3].cpu(), databatch[4].cpu()
        outputs = net(images)
        _, predicted = th.max(outputs, 1)
        print(predicted)
        total += images.shape[0]
        correct += (predicted == categories).sum().item()

print(correct)
print(total)
print(f"Accuracy: {100 * correct / total:.2f}")