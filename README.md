# Deep Learning Specialization

This repository comprises a comprehensive collection of materials for the course of [Deep Learning Specialization, Coursera/DeepLearning.AI](https://www.coursera.org/specializations/deep-learning), including my own solutions to the practice problems and personal notes accumulated during the course. It serves as an invaluable resource for me, aiding in developing a solid foundation for my learning journey in deep learning.

## Description
The Deep Learning Specialization explores fundamental principles in deep learning and is divided into five series courses, each offering thorough explanation and unique insights into this dynamic field. In the following sections, I summarize each course with a brief introduction and illustrative figures.

- [Neural Networks and Deep Learning](https://github.com/lionlai1989/Deep_Learning_Specialization/tree/master/C1-Neural_Networks_and_Deep_Learning):

  The first course explains what are neural networks and what neural networks can solve but conventional algorithms (e.g., linear regression and logistic regression) can't. Moreover, why using **deep** neural networks instead of shallow neural networks is explained and demonstrated with intuitive examples in the Jupyter Notebook.

- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://github.com/lionlai1989/Deep_Learning_Specialization/tree/master/C2-Improving_Deep_Neural_Networks_Hyperparameter_Tuning_Regularization_and_Optimization):

  The second course focuses on various techniques that aid in constructing better neural networks. It covers the significance of random initialization and how it effectively addresses the issues of exploding or vanishing gradients. Additionally, the course explores the normalization of data and hidden layers, various regularization methods including L2 regularization, dropout, and improved ways to perform gradient descent, such as incorporating momentum.

- [Structuring Machine Learning Projects](https://github.com/lionlai1989/Deep_Learning_Specialization/tree/master/C3-Structuring_Machine_Learning_Projects):

  The third course demonstrates techniques for diagnosing errors in machine learning systems, prioritizing strategies to reduce these errors, and understanding complex ML settings. These settings include scenarios such as mismatched training/test sets and comparing or surpassing human-level performance, also known as Bayes optimal error. Moreover, the course presents the details of end-to-end learning, transfer learning, and multi-task learning.

- [Convolutional Neural Networks (CNN)](https://github.com/lionlai1989/Deep_Learning_Specialization/tree/master/C4-Convolutional_Neural_Networks):

  While disregarding the constraints of available training data and computational resources, a fully connected network has the potential to perform all the tasks a CNN can accomplish. This course explains the rationale behind utilizing CNNs to replace fully connected networks. It then covers the foundational concepts necessary to construct a CNN, followed by an introduction to classic networks such as LeNet-5, AlexNet, and VGG-16. Furthermore, the course introduces recent and popular networks, including Residual and Inception networks, and comprehensive explanations of the You Only Look Once (YOLO) algorithm and U-Net.

- Sequence Models (In progress):

## Getting Started
All the results in Jupyter Notebook can be reproduced by following the instructions below.

### Dependencies
Before you start, you need to make sure you have the following dependencies installed:
* **Python-3.10:** Python-3.10 is used throughout all the solutions to the problems. 


### Downloading
* To download this repository, run the following command:
```shell
git clone https://github.com/lionlai1989/Deep_Learning_Specialization.git
```

### Install Python Dependencies
- Create and activate a Python virtual environment
```
python3.10 -m venv venv_deep_learning && source venv_deep_learning/bin/activate
```
- Update `pip` and `setuptools`:
```
python3 -m pip install --upgrade pip setuptools
```
- Install required Python packages in `requirements.txt`.
```
python3 -m pip install -r requirements.txt
```

### Running Jupyter Notebook
Now you are ready to go to each Jupyter Notebook and run the code. Please remember to select the kernel you just created in your virtual environment `venv_deep_learning`.


## Contributing

Any feedback, comments, and questions about this repository are welcome. However, I would like to clarify that this repository serves as a personal record of my learning history, and therefore I will not accept any form of pull requests or merge requests.

## Authors

[@lionlai](https://github.com/lionlai1989)

## Version History

* 0.0.1
    * Finish Deep Learning Specialization in 2023.

<!-- 
Use "python3.10" and "numpy >= 1.20"

## Acknowledgments
Explore the inspiration and references listed here to further expand your knowledge and sharpen your skills.




Coursera: https://learn.udacity.com/courses/ud810

https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml

Find time to do the assignments in the speard sheet above.

https://faculty.cc.gatech.edu/~afb/classes/CS4495-Fall2014/

### NOTE
Make github repository to public so that the images in jupyter notebook can be displayes correctly.


### Installation
Install `sudo apt-get install gfortran` for scipy.

C4W4A1 cannot be build because the model.json file cannot be read. We can build the model and read the weights.

It requires `python-3.7.6` `python-3.10`.
Install environment:  
```
/usr/local/lib/python-3.7.6/bin/python3.7 -m venv venv_deep_learning && source venv_deep_learning/bin/activate && python3 -m pip install --upgrade pip setuptools
```

Install packages:  
```
python3 -m pip install -r requirements.txt
``` -->
