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

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("data_dir", nargs="*", action="store", default="flowers")
# parser.add_argument("save_dir", nargs="*", action="store", default="")

parser.add_argument("--learning_rate", dest="learning_rate", action="store", default=0.01,
                   help="Choose learning rate")
parser.add_argument("--hidden_units", dest="hidden_units", action="store", default=512,
                   help="Choose hidden units of input layer")
parser.add_argument("--epochs", dest="epochs", action="store", default=1,
                   help="Choose number of epochs")
parser.add_argument("--arch", dest="arch", action="store", default="vgg16",
                   help="Choose architecture")
parser.add_argument("--dev", dest="dev", action="store", default="gpu",
                   help="Choose device")

pa = parser.parse_args()

data_dir = pa.data_dir
# save_dir = pa.save_dir

learning_rate = pa.learning_rate
epochs = pa.epochs
hidden_units = pa.hidden_units
arch = pa.arch
dev = pa.dev                   
                              
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if dev == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = dev           
                    
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456,0.406],
                                                            [0.229,0.224,0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

image_datasets = {"train": train_data,
                  "test": test_data,
                  "valid": valid_data}

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

dataloaders = {"train": trainloader,
               "test": testloader,
               "valid": validloader}

# Load .json mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use function to build the model and respective information on
# optimizer, criterion and device
model, optimizer, criterion = tm.nn_builder()

# Send model to device
model.to(device)
steps = 0
print_every = 10
print_count = 0

for e in range(epochs):
    ### Loss tracking ###
    running_loss = 0
    for inputs, labels in dataloaders["train"]:
        print_count += 1

        # Put model in right "mood"
        model.train()
        # Sent data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # (Re)set gradients to zero
        optimizer.zero_grad()

        ### Forward Pass ###
        outputs = model.forward(inputs)
        # Measure the loss
        loss = criterion(outputs, labels)

        ### Backpropagation ###
        loss.backward()
        # Optimize weights using optimizer
        optimizer.step()
        # Capture the loss
        running_loss += loss.item()

            ### Track the error and calculate accuracies ###
        if print_count % print_every == 0:
            validation_loss = 0
            accuracy = 0
            # Put model in evaluation "mood"
            model.eval()

            # turn off gradients f
            with torch.no_grad():
                for inputs, labels in dataloaders["valid"]:
                    # move data to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # feeding
                    outputs = model.forward(inputs) #logps

                    ### Validation Loss and Accuracy ###
                    validation_loss += criterion(outputs, labels)
                    ps = torch.exp(outputs) # exp(outputs) = probabilities
                    top_p, top_class = ps.topk(1, dim=1)
                    # check if equal
                    equals = top_class == labels.reshape(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_loss = running_loss / len(dataloaders["train"])
            validation_loss = validation_loss / len(dataloaders["valid"])
            validation_accuracy = accuracy / len(dataloaders["valid"])

            print("Epoch: {}/{} |".format(e+1, epochs),
                  "Training Loss: {:.3f}|".format(train_loss),
                  "Valid. Loss: {:.3f}|".format(validation_loss),
                  "Valid. Accuracy: {:.3f}".format(validation_accuracy))

            running_loss = 0

# TODO: Save the checkpoint
model.class_to_idx = image_datasets["train"].class_to_idx

# Due to problems when switching between gpu, cpu
#https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
# model.cpu()

checkpoint = {"state_dict": model.state_dict(),
              "class_to_idx":model.class_to_idx,
              "opt_state_dict": optimizer.state_dict()}

torch.save(checkpoint, "checkpoint.pth")

print("End of program: train.py")