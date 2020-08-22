---
layout: default
---

# Getting Started with Machine Learning

[![GitHub stars](https://img.shields.io/github/stars/getting-started-ml/getting-started-ml.github.io?style=social)](https://github.com/getting-started-ml/getting-started-ml.github.io)
[![Buy me a coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/rickwierenga)

A [community-driven](https://github.com/getting-started-ml/getting-started-ml.github.io) place to get started with machine learning and AI. This list is not definite, nor sequential, but we hope it's a good starting place for anyone looking to get into the field. All resources mentioned in this guide are _free_, and include a little description of why they are useful. Each section has a set of starting points (usually courses, books, blog posts, etc.), relevant papers and project ideas. The most helpful starting points in each section are marked with a ⭐️.

_Remember: The two best things you can do are **building stuff** and **running your own experiments**._

## Contents

1. [Basics](#basics)
2. [Mathematics](#mathematics)
   - [Calculus](#calculus)
   - [Linear Algebra](#linear-algebra)
   - [Statistics](#statistics)
3. [Neural Networks](#neural-networks)
4. [Computer Vision](#computer-vision)
5. [Natural Language Processing](#natural-language-processing)
6. [Reinforcement Learning](#reinforcement-learning)
7. [Ethics](#ethics)
8. [Community](#community)
9. [Contributing](#contributing-to-gettingstartedml)

## Basics

- [⭐️ Stanford CS229](http://cs229.stanford.edu): Stanford's introductory machine learning course. Even though the code is quite outdated, this course explains many fundamental ML algorithms. It also has a section on the mathematical foundation required. Additional links: [Python exercises/solutions](https://github.com/rickwierenga/cs229-python) - [Notes](https://stanford.edu/~shervine/teaching/cs-229/) - [Coursera](https://www.coursera.org/learn/machine-learning).
- [Practical Deep Learning for Coders (fast.ai)](https://course.fast.ai): The best course to start with if you are looking to quickly train your own models. This course teaches many clever tricks to improve accuracy and overall performance.
- [Deep Learning Book](http://www.deeplearningbook.org): This book covers everything you need to know to start reading papers. Be sure to write your own code while reading.
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course): Google's fast-paced, practical introduction to machine learning (TensorFlow).

###### Resources

- [Google Colab](https://colab.research.google.com): Free GPUs, shareable notebooks.
- [r/datasets](https://www.reddit.com/r/datasets/): Feed of datasets to fuel your own projects.
- [Dataset Search](https://datasetsearch.research.google.com): A Google for datasets.

## Mathematics

Some beginners are put off by the math. However, machine learning remains a field build upon mathematical principles. While you don't need to know any math to train a neural network, it's quite useful when debugging your architecture, and analyzing model performance. Luckily, the most basic understanding of matrices, derivatives and statistical properties can already help a lot.

- [⭐️ The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html): Digging deeper into the math behind neural networks. Understanding this text will be of tremendous help.
- [Learning Math for Machine Learning](https://www.ycombinator.com/library/51-learning-math-for-machine-learning): Essay on why math is/isn't important in engineering, ML.

### Calculus

- [Calculus Learning Guide](https://betterexplained.com/guides/calculus/): realistic learning plan for Calculus.
- [Essence of calculus by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr): Explore Calculus, the study of change, in an intuitive manner.

### Linear Algebra

- [⭐️ MIT OCW on Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/): An in-depth course on everything you need to know about linear algebra and more. It includes exams with solutions.
- [Essence of linear algebra by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): Build a visual intuition for linear algebra. You probably won't need to know linear transformations, but this course presents some key insights in how everything fits together.
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf): Reference sheet with many key formulas.
- [Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra): Linear Algebra taught with a focus on computing.

### Statistics

- [⭐️ MIT OCW on Statistics](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/index.htm): Interactive, very comprehensive course on statistics. While the video lectures are not available, you can find all readings [here](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/). Each reading has corresponding questions. The exams with solutions are available. This course ends with linear regression: the approximate start of machine learning.
- [probabilitycourse.com](https://www.probabilitycourse.com): Free, online verion of 'Introduction to Probability, Statistics, and Random Processes' by Hossein Pishro-Nik.

## Neural Networks

- [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com): Nice overview of what neural networks can do. Includes Python demos.
- [DeepMind x UCL \| Deep Learning Lecture Series 2020](https://www.youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF): covers many aspects of deep learning.

###### Papers

- [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf): one of the classic ML papers.

###### Project ideas

- Build an CIFAR10 classifier.
- Write a neural network from scratch in Python and NumPy.
- Write a basic NN library with layers, activations functions and a training loop.

## Computer Vision

- [⭐️ Stanford CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu): Industry standard course on computer vision and CNNs.
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf): Nice guide on the mathematical principles behind convolutions. Covers padding, stride, pooling, transposition, etc.

###### Papers

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): One of the first papers on large convolutional networks. Even though it's already outdated at this point, it remains a must-read for anyone looking to understand ConvNets.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

###### Project ideas

- Write an [Imagenette, Imagewoof, Imagewang](https://github.com/fastai/imagenette) classifier.
- Build a model that generates new images of cars.
- Implement ResNet in NumPy, JAX.

## Natural Language Processing

- [⭐️ Stanford CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html): Stanford's undergrad course on NLP. Includes a lot of additional materials.

###### Papers

_Coming soon_

###### Project ideas

_Coming soon_

## Reinforcement Learning

- [⭐️ Spinning Up in Deep RL by OpenAI](https://spinningup.openai.com/en/latest/user/introduction.html): A large collection of RL tutorials by OpenAI, one of the leading AI research institutes.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/first/ebook/the-book.html): Very nice overview of many intuitions, mathematical foundations and algorithms you will encounter in reinforcement learning.
- [RL Course by David Silver (Google DeepMind)](https://www.davidsilver.uk/teaching/): very well organized RL course by one of the creators of AlphaZero.

###### Papers

_Coming soon_

###### Project ideas

- Write a TicTacToe bot.
- Write a chess bot.
- Build a stock trading bot.

## Ethics

Any new field needs to explore its ethical side. Especially AI, because it constantly presents new issues we have never encountered before at an incredible pace. While it may not seem technical, this section is one of the most important if you hope to make an impact with your work.

- [⭐️ Practical Data Ethics](https://ethics.fast.ai): A practical course of fairness, biases, and more.

###### Papers

_Coming soon_

###### Project ideas

- Write a blog post detailing the importance of ethics in AI.
- Explore and report the ethics of real world datasets.

## Community

You can't learn hard things alone. Fortunately, there's a great community ready to help out.

### Resources

- [gettingstarted.ml](http://gettingstarted.ml) ;)
- [paperswithcode.com](https://paperswithcode.com): Machine learning research with corresponding code. A great place to learn to implement literature.

### Discussion

- [r/MachineLearning](https://www.reddit.com/r/machinelearning/): The latest in ML research (advanced).
- [r/LearnMachineLearning](https://www.reddit.com/r/learnmachinelearning/): A subreddit dedicated to learning machine learning.
- [arxiv-sanity.com](http://www.arxiv-sanity.com): An organized directory of the latest ML and stats research papers.
- [Official r/LearnMachineLearning Discord server](https://discord.gg/duHMAGp): Probably the best place to get your short questions answered quickly.

## Contributing to gettingstarted.ml

Thanks for even considering contributing! This guide is community driven, and every [pull request](https://github.com/getting-started-ml/getting-started-ml.github.io/pulls) is appreciated. Whether you are fixing a typo, adding or updating resources - everything will help make this guide better for everyone. Please be careful when submitting your own content, this is not a place for self promotion but we do value good learning materials. On behalf of every beginner, thank you!

---

[Edit on GitHub](https://github.com/getting-started-ml/getting-started-ml.github.io/) &middot; Created by [Rick Wierenga](https://twitter.com/rickwierenga)
