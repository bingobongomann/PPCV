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
        # "car",
        # "plane",
        # "ship",
        # "cat",
        # "horse",
        # "horse",
        # "deer",
        # "frog",
        # "bird",
    ],
    # times=["day"],
    # weathers=["sunny"],
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

label_to_correct_idxes = {
     0: {  # truck
        407,
        555,
        569,
        654,
        675,
        717,
        757,
        779,
        864,
        867,
    },
    1: {  # car
        436,
        468,
        511,
        609,
        627,
        656,
        705,
        734,
        751,
        817,
    },
    2: {  # plane
        404,
        895,
    },
    3: {403, 472, 510, 554, 625, 628, 693, 724, 780, 814, 914},  # ship
    4: {281, 282, 283, 284, 285},  # cat
    5: {  # dog
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        273,
        274,
        275,
    },
    6: {  # "horse"
        339,
        340,
    },
    7: {  # "deer"
        351,
        352,
        353,
    },
    8: {  # "frog"
        30,
        31,
        32,
    },
    9: {  # "bird"
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
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
        total += images.shape[0]

        correct_count = 0
        for pred, cat in zip(predicted.cpu(), categories):
            if int(pred) in list(label_to_correct_idxes[int(cat)]):
                correct_count += 1 

        correct += correct_count

print(correct)
print(total)
print(f"Accuracy: {100 * correct / total:.2f}")