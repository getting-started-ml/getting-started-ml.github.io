---
layout: default
---

# Getting Started with Machine Learning

[![GitHub stars](https://img.shields.io/github/stars/getting-started-ml/getting-started-ml.github.io?style=social)](https://github.com/getting-started-ml/getting-started-ml.github.io)
[![Share on Twitter](https://img.shields.io/twitter/url?url=http://gettingstarted.ml)](https://twitter.com/intent/tweet?text=Looking%20to%20get%20started%20with%20machine%20learning?%20Check%20out%20http://gettingstarted.ml)
[![Buy me a coffee](https://img.shields.io/badge/Support-Buy%20Me%20a%20Coffee-%23EF884F)](https://www.buymeacoffee.com/rickwierenga)

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
5. [Natural Language Processing (NLP)](#natural-language-processing)
6. [Reinforcement Learning (RL)](#reinforcement-learning)
7. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks)
8. [Ethics](#ethics)
9. [Community](#community)
10. [Contributing](#contributing-to-gettingstartedml)

## Basics

- [⭐️ Stanford CS229](http://cs229.stanford.edu): Stanford's introductory machine learning course. Even though the code is quite outdated, this course explains many fundamental ML algorithms. It also has a section on the mathematical foundation required. Additional links: [Python exercises/solutions](https://github.com/rickwierenga/cs229-python) - [Notes](https://stanford.edu/~shervine/teaching/cs-229/) - [Coursera](https://www.coursera.org/learn/machine-learning).
- [Practical Deep Learning for Coders (fast.ai)](https://course.fast.ai): The best course to start with if you are looking to quickly train your own models. This course teaches many clever tricks to improve accuracy and overall performance.
- [Deep Learning Book](http://www.deeplearningbook.org): This book covers everything you need to know to start reading papers. Be sure to write your own code while reading.
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course): Google's fast-paced, practical introduction to machine learning (TensorFlow).
- [Dive into Deep Learning](http://d2l.ai/index.html): An interactive deep learning book with code, math, and discussions. Provides NumPy/MXNet, PyTorch, and TensorFlow implementations.

###### Resources

- [Google Colab](https://colab.research.google.com): Free GPUs, shareable notebooks.
- [r/datasets](https://www.reddit.com/r/datasets/): Feed of datasets to fuel your own projects.
- [Dataset Search](https://datasetsearch.research.google.com): A Google for datasets.

## Mathematics

- [⭐️ The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html): Digging deeper into the math behind neural networks. Understanding this text will be of tremendous help.
- [Learning Math for Machine Learning](https://www.ycombinator.com/library/51-learning-math-for-machine-learning): Essay on why math is/isn't important in engineering, ML.
- [A Programmer's Introduction to Mathematics](https://pimbook.org): Be sure to read this if you have never read a college level math book before. It not only teaches fundamental math, but also introduces the reader to succesfully reading mathematical literature.

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
- [Theoretical issues in deep networks](https://www.pnas.org/content/early/2020/06/08/1907369117)

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

_Coming soon_ - [Open a PR!](https://github.com/getting-started-ml/getting-started-ml.github.io)

###### Project ideas

_Coming soon_ - [Open a PR!](https://github.com/getting-started-ml/getting-started-ml.github.io)

## Reinforcement Learning

- [⭐️ Spinning Up in Deep RL by OpenAI](https://spinningup.openai.com/en/latest/user/introduction.html): A large collection of RL tutorials by OpenAI, one of the leading AI research institutes.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/first/ebook/the-book.html): Very nice overview of many intuitions, mathematical foundations and algorithms you will encounter in reinforcement learning.
- [RL Course by David Silver (Google DeepMind)](https://www.davidsilver.uk/teaching/): very well organized RL course by one of the creators of AlphaZero.

###### Papers

_Coming soon_ - [Open a PR!](https://github.com/getting-started-ml/getting-started-ml.github.io)

###### Project ideas

- Write a TicTacToe bot.
- Write a chess bot.
- Build a stock trading bot.

## Generative adversarial networks

* [DeepMind: Generative Adversarial Networks](https://www.youtube.com/watch?v=wFsI2WqUfdA&list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF&index=10)
* Lectures [5](https://www.youtube.com/watch?v=1CT-kxjYbFU&list=PLwRJQ4m4UJjPiJP3691u-qWwPGVKzSlNP&index=5) and [6](https://www.youtube.com/watch?v=0W1dixJfKL4&list=PLwRJQ4m4UJjPiJP3691u-qWwPGVKzSlNP&index=6) of Berkeley's Deep Unsupervised Learning.

###### Papers

* [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
* [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
* [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
* [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
* [Improved Techniques for Training GANs](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)
* [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
* [cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)
* [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
* [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
* [BigGAN](https://arxiv.org/abs/1809.11096)
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196), [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) and [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

## Ethics

- [⭐️ Practical Data Ethics](https://ethics.fast.ai): A practical course of fairness, biases, and more.
- [CS 294: Fairness in Machine Learning](https://fairmlclass.github.io): List of good readings on fairness.
- [21 fairness definitions and their politics](https://fairmlbook.org/tutorial2.html): Talk from the [FAT](https://facctconference.org) conference.
- [NIPS 2017 Tutorial on Fairness in Machine Learning](https://fairmlbook.org/tutorial1.html): Presents a 'toolkit' researchers and engineers can use to improve fairness.

###### Papers

- [Ethics of Artificial Intelligence and Robotics](https://plato.stanford.edu/entries/ethics-ai/)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
- [A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/abs/1908.09635)

###### Project ideas

- Write a blog post detailing the importance of ethics in AI.
- Explore and report the ethics of real world datasets.

## Community

You can't learn hard things alone. Fortunately, there's a great community ready to help out.

### Resources

- [gettingstarted.ml](http://gettingstarted.ml) ;)
- [paperswithcode.com](https://paperswithcode.com): Machine learning research with corresponding code. A great place to learn to implement literature.

### Blogs / publications

- [distill.pub](https://distill.pub): interactive articles about new machine learning research, or new viewpoints on existing work.
- [OpenAI Blog](https://openai.com/blog/)
- [DeepMind Blog](https://deepmind.com/blog)
- [Google AI Blog](https://blog.google/technology/ai/)
- [Colah's Blog](https://colah.github.io): Very good explanations of intermediate ML topics.

### Discussion

- [r/MachineLearning](https://www.reddit.com/r/machinelearning/): The latest in ML research (advanced).
- [r/LearnMachineLearning](https://www.reddit.com/r/learnmachinelearning/): A subreddit dedicated to learning machine learning.
- [arxiv-sanity.com](http://www.arxiv-sanity.com): An organized directory of the latest ML and stats research papers.
- [Official r/LearnMachineLearning Discord server](https://discord.gg/duHMAGp): Probably the best place to get your short questions answered quickly.

###### Papers

- [Ten simple rules for getting started on Twitter as a scientist](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1007513&type=printable)

## Contributing to gettingstarted.ml

Thanks for even considering contributing! This guide is community driven, and every [pull request](https://github.com/getting-started-ml/getting-started-ml.github.io/pulls) is appreciated. Whether you are fixing a typo, adding or updating resources - everything will help make this guide better for everyone. Please be careful when submitting your own content, this is not a place for self promotion but we do value good learning materials. On behalf of every beginner, thank you!

---

[Edit on GitHub](https://github.com/getting-started-ml/getting-started-ml.github.io/) &middot; Created by [Rick Wierenga](https://twitter.com/rickwierenga)
