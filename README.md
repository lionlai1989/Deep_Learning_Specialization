# Deep Learning Specialization

This repository comprises a comprehensive collection of materials for the course of [Deep Learning Specialization, Coursera/DeepLearning.AI](https://www.coursera.org/specializations/deep-learning), including my own solutions to the practice problems and personal notes accumulated during the course. It serves as an invaluable resource for me, aiding in developing a solid foundation for my learning journey in deep learning.

## Description
The Deep Learning Specialization explores fundamental principles in deep learning and is divided into five series courses, each offering thorough explanation and unique insights into this dynamic field. In the following sections, I summarize each course with a brief introduction and illustrative figures.

- [Neural Networks and Deep Learning](https://github.com/lionlai1989/Deep_Learning_Specialization/tree/master/C1-Neural_Networks_and_Deep_Learning):

  The first course explains what are neural networks and what neural networks can solve but conventional algorithms (e.g., linear regression and logistic regression) can't. Moreover, why using **deep** neural networks instead of shallow neural networks is explained and demonstrated with intuitive examples in the Jupyter Notebook.

- Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization:

  The second course focuses on various techniques that aid in constructing better neural networks. It covers the significance of random initialization and how it effectively addresses the issues of exploding or vanishing gradients. Additionally, the course explores the normalization of data and hidden layers, various regularization methods including L2 regularization, dropout, and improved ways to perform gradient descent, such as incorporating momentum.

- Structuring Machine Learning Projects:

  The third course demonstrates techniques for diagnosing errors in machine learning systems, prioritizing strategies to reduce these errors, and understanding complex ML settings. These settings include scenarios such as mismatched training/test sets and comparing or surpassing human-level performance, also known as Bayes optimal error. Moreover, the course presents the details of end-to-end learning, transfer learning, and multi-task learning.

- Convolutional Neural Networks (CNN):

  While disregarding the constraints of available training data and computational resources, a fully connected network has the potential to perform all the tasks a CNN can accomplish. This course explains the rationale behind utilizing CNNs to replace fully connected networks. It then covers the foundational concepts necessary to construct a CNN, followed by an introduction to classic networks such as LeNet-5, AlexNet, and VGG-16. Furthermore, the course introduces recent and popular networks, including Residual and Inception networks, and comprehensive explanations of the You Only Look Once (YOLO) algorithm and U-Net.

- Sequence Models:


Use "python3.10" and "numpy >= 1.20"

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

Your feedback, comments, and questions about this project are welcome, as are any contributions you'd like to make. Please feel free to create an issue or a pull request in this repository. Together, let's improve this template and make our life easier.

## Authors

[@lionlai](https://github.com/lionlai1989)

## Version History

* 0.0.1
    * Finish Deep Learning Specialization in 2023.

## Acknowledgments
Explore the inspiration and references listed here to further expand your knowledge and sharpen your skills.




Coursera: https://learn.udacity.com/courses/ud810

https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml

Find time to do the assignments in the speard sheet above.

https://faculty.cc.gatech.edu/~afb/classes/CS4495-Fall2014/

### NOTE
Make github repository to public so that the images in jupyter notebook can be displayes correctly.


### Some useful references in each course.
- Course 1:
  - [Implementing a Neural Network from Scratch in Python – An Introduction](https://github.com/dennybritz/nn-from-scratch)
  - [Why normalize images by subtracting dataset's image mean, instead of the current image mean in deep learning?](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)
  - [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/neural-networks-case-study/)
  - [Autoreload of modules in IPython](https://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython)


- Course 4:
  - Week 1:
    - [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model) (TensorFlow Documentation)
    - [The Functional API](https://www.tensorflow.org/guide/keras/functional) (TensorFlow Documentation)

  - Week 2:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren & Sun, 2015)
    - [deep-learning-models/resnet50.py/](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py) (GitHub: fchollet)
    - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, & Adam, 2017)
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (Sandler, Howard, Zhu, Zhmoginov &Chen, 2018)
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)

  - Week 3:
    - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon, Divvala, Girshick & Farhadi, 2015)
    - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (Redmon & Farhadi, 2016)
    - [YAD2K](https://github.com/allanzelener/YAD2K) (GitHub: allanzelener)
    - [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
    - [Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs](https://arxiv.org/abs/1701.08816) (Novikov, Lenis, Major, Hladůvka, Wimmer & Bühler, 2017)
    - [Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks](https://arxiv.org/abs/1705.03820) (Dong, Yang, Liu, Mo & Guo, 2017)
    - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer & Brox, 2015)

  - Week 4:
    - [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf) (Schroff, Kalenichenko & Philbin, 2015)
    - [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) (Taigman, Yang, Ranzato & Wolf)
    - [facenet](https://github.com/davidsandberg/facenet) (GitHub: davidsandberg)
    - [How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) (Jason Brownlee, 2019)
    - [keras-facenet/notebook/tf_to_keras.ipynb](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb) (GitHub: nyoki-mtl)
    - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys, Ecker & Bethge, 2015)
    - [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
    - [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
    - [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) (Simonyan & Zisserman, 2015)
    - [Pretrained models](https://www.vlfeat.org/matconvnet/pretrained/) (MatConvNet)


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
```
