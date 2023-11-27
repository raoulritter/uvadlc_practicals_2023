  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import matplotlib.pyplot as plt
import cifar10_utils
from train_mlp_numpy import confusion_matrix as np_confusion_matrix, confusion_matrix_to_metrics as np_confusion_matrix_to_metrics


import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    conf_mat = np_confusion_matrix(predictions, targets)
      

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    metrics = np_confusion_matrix_to_metrics(confusion_matrix, beta=beta)

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    ######################
    beta_list = [0.1, 1, 10]
    model.eval()
    data_y = []
    data_prob = []
    device = torch.device('mps') if torch.backends.mps.is_available()  else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    confusion_mat = np.zeros((num_classes, num_classes))
    for image, label in data_loader:
        x, y = image.to(device), label.to(device)
        with torch.no_grad():
          preds = model(x)
          data_y.extend(y.tolist())
          data_prob.extend(preds.tolist())
    confusion_mat += confusion_matrix(preds.cpu().numpy(), y.cpu().numpy())
    metrics = confusion_matrix_to_metrics(confusion_mat, beta=1)
    for beta in beta_list:
        metrics = confusion_matrix_to_metrics(confusion_mat, beta=beta)
 




    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
      torch.mps.manual_seed(seed)
      device = torch.device('mps')
    else:
      device = torch.device('cpu')


    # Set default device
    # device = torch.device('cuda' if torch.cuda.is_available() elif torch.backends.mps.is_available() else 'cpu')

    # Set default device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    image_size = 3 * 32 * 32

    class_count = 10


    best_accuracy = 0
    best_model = None
    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []


    # # TODO: Initialize model and loss module
    model = MLP(image_size, hidden_dims, class_count, use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()


    # # TODO: Training loop including validation\       
   

    model = MLP(image_size, hidden_dims, 10).to(device)

    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(cifar10_loader['train']):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_module(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Accumulate the loss

        train_losses.append(train_loss / len(cifar10_loader['train']))
        # train_metrics = evaluate_model(model, cifar10_loader['train'], 10)
        # train_accuracies.append(train_metrics['accuracy'])

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, labels in cifar10_loader['validation']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_module(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(cifar10_loader['validation']))
        val_accuracy = evaluate_model(model, cifar10_loader['validation'], 10)['accuracy']
        val_accuracies.append(val_accuracy)
        test_metrics  = evaluate_model(model, cifar10_loader['test'], 10)
        test_accuracies.append(test_metrics['accuracy'])
        

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = deepcopy(model)
            # Save your best model
            torch.save(best_model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]}, Validation Accuracy: {val_accuracy}')

     # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'], 10)['accuracy']
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Highest test accuracy:{max(test_accuracies)}')
    # Model appears to be overfitting, as the test accuracy is lower than the validation accuracy.

    logging_info = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
    }
 

    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_pytorch.png')
    plt.show()  


    plt.figure(figsize=(10, 8))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_pytorch.png')
    plt.show()


    


    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    