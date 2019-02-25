# ImageClassifierProject
Deep Learning Project to classify Images using PyTorch, Neural nets, GD, and transfer learning with GPU

## Part 1: ImageClassifierProject.ipynb
Contents: 
- Introduce, load and preprocess image dataset
-- `transforms.`, `ImageFolder`, `Dataloader`
- Label mapping  
 - Building and training the classifier
  - Function to build a Neural Network and use pretrained model (e.g. vgg16, densenet121).
 - Batch Gradient Descent to optimize classifier using GPU Power
- Testing the trained network, saving and loading checkpoints
- Inference for classification
 - Function to predict class of an Image
 - Sanity checking

## Part 2: train.py, predict.py
- Develop an AI application which can be used from the command line
 
Function 1: train.py 
- Trains a neural network classifier. 

Function 2: predict.py 
- Predicts likelihood that Image belongs to a certain category trained with train.py

## Other files:
- tm_fun.py: some helper functions for part 2
- cat_to_name.json: Consists of the mapping of categories to flower names
- workspace-utils.py: Keeps workspace awake if GPU takes longer...
