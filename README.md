# Deep Learning Specialization
This GitHub repository contains all the necessary materials for the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning). It encompasses my own solutions to the practice problems, along with my personal notes that I have taken throughout the course. This repository serves as an invaluable resource for me, aiding in the development of a strong foundation in deep learning skills.

## Description
The Deep Learning Specialization explores fundamental principles in deep learning and is divided into five series courses, each offering comprehensive explanation and unique insights into this dynamic field. In the following sections, I summarize each course with a brief introduction and illustrative figures.

- Neural Networks and Deep Learning:

  The first course explains why we need neural networks and what neural networks can solve but conventional algorithms (e.g., linear regression and logistic regression) can't. Moreover, why using **deep** neural networks instead of shallow neural networks is explained and demonstrated with intuitive examples in the Jupyter Notebook.

- Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization:

  The second course focuses on various techniques that aid in constructing better neural networks. It covers the significance of random initialization and how it effectively addresses the issues of exploding or vanishing gradients. Additionally, the course explores the normalization of data and hidden layers, various regularization methods including L2 regularization, dropout, and improved ways to perform gradient descent, such as incorporating momentum.

- Structuring Machine Learning Projects:

  The third course demonstrates techniques for diagnosing errors in machine learning systems, prioritizing strategies to reduce these errors, and understanding complex ML settings. These settings include scenarios such as mismatched training/test sets and comparing or surpassing human-level performance, also known as Bayes optimal error. Moreover, the course covers the application of end-to-end learning, transfer learning, and multi-task learning.

- Convolutional Neural Networks (CNN):

  While disregarding the constraints of available training data and computational resources, a fully connected network has the potential to perform all the tasks a CNN can accomplish. This course explains the rationale behind utilizing CNNs to replace fully connected networks. It then covers the foundational concepts necessary to construct a CNN, followed by an introduction to classic networks such as LeNet-5, AlexNet, and VGG-16. Furthermore, the course introduces recent and popular networks, including Residual and Inception networks, and comprehensive explanations of the You Only Look Once (YOLO) algorithm and U-Net.

- Sequence Models:


Use "python3.10" and "numpy >= 1.20"

## Getting Started
All the results can be reproduced by following the instructions below.

### Dependencies
Before you start, you need to make sure you have the following dependencies installed:
* **CMake 3.22.0 or higher:** If you don't have CMake installed, or if you need to update it, you can follow the instructions [here](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line). To use version 3.22, you can download it from https://cmake.org/files/v3.22/cmake-3.22.6.tar.gz.
* **Eigen library:** This is a C++ library that we'll use in our project. Don't worry about installing it separately, as it's included as a git submodule in our repository.
* **stb image library:** This is a C library for loading and saving images. It's also included as a git submodule, so you don't need to do anything extra.
* **xtensor-assosiated library:** xtensor is a numpy for C++ library. All
  required code is added as git submodules. Unlike eigen and stb library, I
  can't figure out a way to just add folders to make CMake work. We need to
  first install xtensor and then use it. Please follow the instruction below:
  * Isntall xtl:
  ```
  (cd extern/xtl && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xtl-install && make install)
  ```
  * Install xtensor:
  ```
  (cd extern/xtensor && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xtensor-install -DCMAKE_PREFIX_PATH=/tmp/xtl-install && make install)
  ```
  * Install xsimd:
  ```
  (cd extern/xsimd && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xsimd-install && make install)
  ```
  

* **Development tools for Linux and VS Code:** To develop our project, we'll be using Linux and Visual Studio Code (VS Code). To have a smoother experience, you should install the following tools and extensions for VS Code:
  * `C/C++`
  * `C/C++ Extension`
  * `CMake`
  * `CMake Extension`
  * `llvm`
  * `lldb`
  * `ninja`
  * `python3-dev` (install this with apt install python3-dev)

### Downloading
* To download this repository, run the following command:
```shell
git clone --recursive https://github.com/lionlai1989/cmake_template.git
```
If you forgot to use the `--recursive` option when cloning, you can still clone the submodules by running the command `git submodule update --init --recursive`.

### Build, Install and Execute
- Build:  
  To build the project, run the following command:
  ```
  cmake -G Ninja -S . -B build/ && cmake --build build/ -j 4 && (cd build/; ctest -V)
  ```
  If the build is successful, the tests will be run automatically and you should see the message:
  ```
  100% tests passed, 0 tests failed out of 5
  ```

- Install  
  This package can be installed in your system or a custom location in your file system. To install it, run the following command:
  ```
  cmake --install build/ --prefix /tmp/install-test/
  ```
  The above code installs the package in `/tmp/install-test/`.
  
- Execute  
  The installation can be tested with the following command:
  ```
  /tmp/install-test/bin/rgb2gray -i /tmp/install-test/bin/book_in_scene.jpg -o ./examples/files/book_in_scene_gray.jpg -m eigen
  /tmp/install-test/bin/rgb2gray -i /tmp/install-test/bin/book.png -o ./examples/files/book_gray.png -m xtensor
  ```
  It will use `eigen` to create `book_in_scene_gray.jpg` and `xtensor` to create `book_gray.png` in `./examples/files/`. Here is an example input and output images.
  <p align="left">
    <img src="./examples/files/book_in_scene.jpg" width="300" title="Input RGB Image with JPG Format">
    <img src="./examples/files/book_in_scene_gray.jpg" width="300" title="Output Grayscale Image with JPG Format">
  </p>
  <p align="left">
    <img src="./examples/files/book.png" width="300" title="Input RGB Image with PNG Format">
    <img src="./examples/files/book_gray.png" width="300" title="Output Grayscale Image with PNG Format">
  </p>

### Developing
- Using Libraries in this Package:  
  To use the libraries included in this package, it is necessary to include the appropriate headers in your code and link to the libraries in your project.

- Future Developments:  
  In the future, there are plans to expand this project to include additional example libraries to further explore their use. These libraries include libvips, CImg, terrasect, and opencv. Additionally, the code will be further improved by utilizing tools such as Clang Static Analyzer and clang-tidy to identify potential issues and enhance overall code quality.

## Contributing

Your feedback, comments, and questions about this project are welcome, as are any contributions you'd like to make. Please feel free to create an issue or a pull request in this repository. Together, let's improve this template and make life easier for C++ programmers.

## Authors

[@lionlai](https://github.com/lionlai1989)

## Version History

* 0.0.1
    * Initial Release

## Acknowledgments
Explore the inspiration and references listed here to further expand your knowledge and sharpen your skills.




Coursera: https://learn.udacity.com/courses/ud810

https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml

Find time to do the assignments in the speard sheet above.

https://faculty.cc.gatech.edu/~afb/classes/CS4495-Fall2014/


# installation
```
python3.10 -m venv venv_deep_learning && source venv_deep_learning/bin/activate && python3 -m pip install --upgrade pip setuptools && python3 -m pip install -r requirements.txt
```

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
