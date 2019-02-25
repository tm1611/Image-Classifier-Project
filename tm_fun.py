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

def nn_builder(mod = "vgg16", drop_rate=0.2, hidden_units = 512, learnrate=0.001):
    '''
    This function builds a simple neural network, where the features are taken from
    either vgg16 or densenet121.
    The classifier consists of one hidden layer with ReLU, specifiable number of hidden_units and dropout.
    Criterion and optimizer are hardcoded to keep this function simple.
    The function could be easily be modified to account for other architectures,
    optimizers, criteria, etc. if desired.
    '''
    if mod == "vgg16":
        model = models.vgg16(pretrained=True)
    elif mod == "vgg13":
        model = models.vgg13(pretrained=True)
    elif mod == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        print("Choose either vgg16 or densenet121")

    num_filters = model.classifier[0].in_features
    # freeze parameters (avoid backprop)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(num_filters, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(drop_rate),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model, optimizer, criterion

def rebuild(pth):
    checkpoint = torch.load(pth)
    model,_,_ = nn_builder()
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["class_to_idx"]

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
    # Compose as before.
    adjustments = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
    ])
    img_tensor = adjustments(img_pil)
    return img_tensor

def predict(input_img, model, topk=5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_to_idx = model.class_to_idx
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    
    model.to(device)
    img = process_image(input_img)
    img = img.unsqueeze_(0)
    img = img.float()

    with torch.no_grad():
        ps = torch.exp(model(img.to(device)))
        top_p, top_idx = ps.topk(topk, dim=1)
    probs = np.array(top_p[0])
    top_idx = np.array(top_idx[0])
    # Convert idx to top classes
    classes = []
    for idx in top_idx:
        classes += [idx_to_class[idx]]

    return probs, classes
