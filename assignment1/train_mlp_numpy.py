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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import matplotlib.pyplot as plt
import cifar10_utils

import torch


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

    class_count = predictions.shape[1]   
    conf_mat = np.zeros(shape=(10, 10))
    for sample, label in zip(predictions, targets):
        prediction = np.argmax(sample)
        conf_mat[label][prediction] += 1

    return conf_mat
      


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


    
    n_classes = 10
    metrics = {}
    metrics['accuracy'] = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    metrics['precision'] = np.zeros(n_classes)
    metrics['recall'] = np.zeros(n_classes)

    
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        metrics['precision'][i] = tp / (tp + fp)
        metrics['recall'][i] = tp / (tp + fn)
        metrics['accuracy'] = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        metrics['f1_beta'] = (1 + beta ** 2) * metrics['precision'] * metrics['recall'] / (beta ** 2 * metrics['precision'] + metrics['recall'])
        metrics.update({'precision': metrics['precision'], 'recall': metrics['recall'], 'f1_beta': metrics['f1_beta'], 'beta': beta}, )

    #Metrics in df table
    # Uncomment to save metrics but ugly
    # metrics_df = pd.DataFrame(metrics)
    # metrics_df.index = np.arange(0, 10) # Classes from 1 to 10
    # #df nan to 0
    # #The first column should title Class and the first row should be titled Metric
    # metrics_df = metrics_df.fillna(0)
    # print(metrics_df)
    # metrics_df.to_csv('metrics.csv')    
    # plt.figure(figsize=(10, 8))
    # plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.savefig('confusion_matrix.png')

    # print('Accuracy: ', metrics['accuracy'])
    
    #######################

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
    #######################

    probability = []
    y_labels = []
    for batch in data_loader:
      x, y = batch
      prob = model.forward(x.reshape(x.shape[0], -1))
      y_labels.extend(y.tolist())
      probability.extend(prob.tolist())
    
    conf_mat = confusion_matrix(np.array(probability), np.array(y_labels))
    metrics = confusion_matrix_to_metrics(conf_mat)

    

    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################


    # Unpack the data loaders
    test_loader = cifar10_loader["test"]

    # Initialize the model and the loss module
    data_size = 32 * 32 * 3
    num_classes = 10
    model = MLP(data_size, hidden_dims, num_classes)
    loss_module = CrossEntropyModule()

    # Prepare arrays to store metrics
    train_accuracies = []
    test_accuracy = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = 0
        total_train_samples = 0

        for images, labels in tqdm(cifar10_loader["train"]):
            images = np.reshape(images, newshape=(images.shape[0], data_size))
            predictions = model.forward(images)
            loss = loss_module.forward(predictions, labels)

            dout = loss_module.backward(predictions, labels)
            model.backward(dout)

            for layer in model.layers:
                if hasattr(layer, "params"):
                    layer.params["weight"] -= lr * layer.grads["weight"]
                    layer.params["bias"] -= lr * layer.grads["bias"]

            train_loss += loss * len(images)
            total_train_samples += len(images)

        train_losses.append(train_loss / total_train_samples)
        train_metrics = evaluate_model(model, cifar10_loader["train"])
        train_accuracies.append(train_metrics["accuracy"])

        val_loss = 0
        total_val_samples = 0
        for images, labels in tqdm(cifar10_loader["validation"]):
            images = np.reshape(images, newshape=(images.shape[0], data_size))
            predictions = model.forward(images)
            loss = loss_module.forward(predictions, labels)

            val_loss += loss * len(images)
            total_val_samples += len(images)

        val_losses.append(val_loss / total_val_samples)
        val_metrics = evaluate_model(model, cifar10_loader["validation"])
        val_accuracies.append(val_metrics["accuracy"])
        test_metrics = evaluate_model(model, cifar10_loader["test"])
        test_accuracy.append(test_metrics["accuracy"])

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_model = deepcopy(model)
            
        model.clear_cache()

    print(f'Test Accuracy: {test_accuracy}')

    logging_dict = {
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_accuracy": test_accuracy,
        "best_model": best_model if best_model else model,
        "best_accuracy": best_accuracy,
    }


    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_np.png')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(test_accuracy, label='Test accuracy')
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_np.png')
    plt.show()


    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict





if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
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

    
    