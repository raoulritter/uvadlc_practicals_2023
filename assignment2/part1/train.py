################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from cifar100_utils import get_train_validation_set, get_test_set, add_augmentation


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(torch.device)


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.normal_(resnet18.fc.weight, mean=0, std=0.01)
    nn.init.zeros_(resnet18.fc.bias)
    model = resnet18

    #######################
    # END OF YOUR CODE    #
    #######################

    return model



def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None, ):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # # Set model to training mode and move to the device.
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True # only train the last layer
    # # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cross_entropy_loss = nn.CrossEntropyLoss()

    # # Training loop with validation after each epoch. Save the best model.
    best_acc = 0.0
    print('Training...')
    print('epochs: {}'.format(epochs))
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

        val_acc = evaluate_model(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = best_acc
            torch.save(model.state_dict(), checkpoint_name)
            print(f'epoch: {epoch}, val_acc: {val_acc}')
    model.load_state_dict(torch.load(checkpoint_name))

    # # Training loop with validation after each epoch. Save the best model.




    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    correct = 0
    total = 0
    # Loop over the dataset and compute the accuracy. Return the accuracy
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(dim=-1)==labels).sum().item()
            total += labels.shape[0]


    accuracy = correct / total

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the model
    model = get_model()
    model.to(device)

    augmentation = None

    # Get the augmentation to use
    if augmentation_name is not None:
        transform_list = []
        augmentation = add_augmentation(augmentation_name, transform_list)
        print("Added augmentation: ", augmentation_name)

    # Train the model
    # model = train_model(model, lr, batch_size, epochs, data_dir, "model_augment.pth", device, augmentation)
    def load_checkpoint(model, checkpoint_name, device):
        model.load_state_dict(torch.load(checkpoint_name))
        model.to(device)
        return model
    model = load_checkpoint(model, "model.pth", device)



    # load the best model
    model.load_state_dict(torch.load("model_augment.pth")) if augmentation_name is not None else model.load_state_dict(torch.load("model.pth"))
    print("Done training!")

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir, test_noise)
    print(test_noise, augmentation_name)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_acc = evaluate_model(model, test_loader, device)

    print(f"Accuracy on the test set: {test_acc * 100:.2f}%")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
