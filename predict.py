# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import numpy as np
from PIL import Image

import argparse
import tm_fun as tm

parser = argparse.ArgumentParser(description="predict.py")

parser.add_argument("checkpoint", nargs="*", action="store", default="checkpoint.pth",
                   help = "Name and place of checkpoint")
parser.add_argument("input_img", nargs="*", action="store", type=str, default="flowers/test/11/image_03165.jpg",
                   help = "Choose path of the image you want to be predicted")
parser.add_argument("category_names", nargs="*", action="store", type=str, default="cat_to_name.json",
                   help="input category names as .json")
parser.add_argument("--topk", dest="topk", action="store", default=1,
                   help="Decide how many top k flowers shall be shown")
parser.add_argument("--dev", dest="dev", action="store", default="gpu",
                   help="GPU or CPU")

pa = parser.parse_args()
checkpoint = pa.checkpoint
input_img = pa.input_img
topk = pa.topk
cat_to_name = pa.category_names
dev = pa.dev

if dev == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = dev    

# Load .json mapping
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

# Rebuild the model
model, class_to_idx = tm.rebuild(checkpoint)
process_image = tm.process_image

# class_to_idx and idx_to_class
class_to_idx = model.class_to_idx
idx_to_class = {x: y for y, x in class_to_idx.items()}

# Predict the Image
probs, classes = tm.predict(input_img, model, topk)
print(probs)
print(classes)

# Convert class to cat
flower_names = []
for i in classes:
    flower_names += [cat_to_name[i]]

print("Based on the given picture, the most likely flower/s are: ")

for i in range(len(flower_names)):
    p = probs[i]
    print(i+1, flower_names[i].title(),"with prob.: {:.2f}%".format(p*100))

print("End of program: predict.py")