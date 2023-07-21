# Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

In this course, I learn the best practices for splitting training and developing test sets, analyzing bias/variance when building deep learning applications. I also learn the standard neural network techniques, including initialization, L2 and dropout regularization, hyperparameter tuning, batch normalization, and gradient checking. Moreover, I apply various optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop, and Adam. Finally, I familiarize myself with TensorFlow by building a neural network.  

## Description

This section provides a concise summary of each assignment in the course, accompanied by brief descriptions and a few figures.

- W2A1: [Initialization](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization/W1A1-Initialization/Initialization.html)  
Initialization plays an important role in training neural networks. Zero initialization fails to break the symmetry, leading to no reduction in the cost. On the other hand, initializing weights and biases randomly does break the symmetry, but it requires more time to optimize the network. However, using He initialization demonstrates that it achieves the lowest cost in the least amount of time.  

- W2A2: [Regularization](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization/W1A2-Regularization/Regularization.html)  
Neural networks without regularization tend to overfit the training data, resulting in decreased performance during inference time. Applying L2 regularization and dropout techniques helps prevent neural networks from exhibiting high variance, thus improving generalization and robustness.  

<figure float="left">
<img src="./W1A2-Regularization/experiment_output/nn_no_regularization.png" width="300"/>
<img src="./W1A2-Regularization/experiment_output/nn_l2.png" width="300"/>
<img src="./W1A2-Regularization/experiment_output/nn_dropout.png" width="300"/>
</figure>

- W1A3: [Gradient Checking](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization/W1A3-Gradient_Checking/Gradient_Checking.html)  
Imagine yourself as Geoffrey Hinton back in 2012 when you were implementing backpropagation on a neural network for the first time. How did you ensure that your implementation was correct? Like any other research, how could you be sure about the result if it had never been done before? Gradient checking provided a way to verify the accuracy of your backpropagation.  

- W2A1: [Optimization Methods](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization/W2A1-Optimization_Methods/Optimization_methods.html)  
Optimization methods, such as mini-batch, RMSProp, Adam, etc., almost always improve the training process. This report provides intuitive visualizations and code snippets to demonstrate how these optimizations are conducted and influence performance during inference.  

- W3A1: [Introduction_to_Tensorflow](https://htmlpreview.github.io/?https://github.com/lionlai1989/Deep_Learning_Specialization/blob/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization/W3A1-Introduction_to_Tensorflow/Introduction_to_Tensorflow.html)  
Until now, I've been implementing all the mechanics of neural networks with NumPy without using any machine learning frameworks. In this assignment, I will familiarize myself with TensorFlow since it will be used throughout this specialization. It's unfortunate that this course doesn't choose PyTorch, considering that PyTorch surpassed TensorFlow in 2023.  

## Reference:

- Week 3:
  - [Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff) (TensorFlow Documentation)
  - [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) (TensorFlow Documentation)
