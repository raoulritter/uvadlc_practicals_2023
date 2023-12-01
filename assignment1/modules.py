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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        #Initialize weight parameters using Kaiming initialization

        self.params = {
          'weight': np.random.normal(0, np.sqrt(2/in_features), (out_features, in_features)),
          'bias': np.zeros((1, out_features)),
        }

        self.grads = {
          'weight': np.zeros(self.params['weight'].shape),
          'bias': np.zeros(self.params['bias'].shape),
        }


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x  # Store input for backward pass

        out = x @ self.params['weight'].T + self.params['bias'] 

        self.output = out 

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Gradient of weight
        self.grads['weight'] = dout.T @ self.input

        # Gradient of bias from the sum of the gradients of the loss with respect to the output
        self.grads['bias'] = np.sum(dout, axis=0)

        
        dx = dout @ self.params['weight']

        




        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #Set any caches you have to None.
        # self.x = None

        self.input = None
        self.output = None
        self.dout = None
        self.dx = None
        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #Store intermediate variables inside the object. They can be used in backward pass computation.
        self.input = x

        out = np.where(x > 0, x, np.exp(x) - 1)



        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        dx = np.where(self.input > 0, dout, dout * np.exp(self.input))

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.input = None
        self.output = None
        self.dout = None
        self.dx = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #Copy Paste from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        self.input = x
        b = x.max(axis=1, keepdims=True)
        y = np.exp(x - b)
        out = y / y.sum(axis=1, keepdims=True)
        self.output = out # Store output for backward pass

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """


        #######################
        # PUT YOUR CODE HERE  #
        #######################

      


        identity = np.eye(self.output.shape[1]) # Identity matrix based off of the number of classes

        diagonal = np.einsum('ij,jk->ijk', self.output, identity) # Diagonal matrix with softmax output on the diagonal
        outer_product = np.einsum('ij,ik->ijk', self.output, self.output) # Outer product of softmax output
  
        jacobian_batch = diagonal - outer_product

        dx = np.einsum('ij,ijk->ik', dout, jacobian_batch)
        

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.input = None
        self.output = None
        self.dout = None
        self.dx = None



        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.input = x
        num_class = x.shape[1]
        y_one_hot = np.eye(num_class)[y]

        self.labels = y_one_hot


        #calculate cross entropy loss
        out = -np.sum(y_one_hot * np.log(x)) / x.shape[0]

        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #calculate gradient of the loss with the respect to the input x.
        num_class = x.shape[1]
        y_one = np.eye(num_class)[y]
        dx = -(y_one / (x)) / x.shape[0]

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx