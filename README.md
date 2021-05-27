
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>
    
--------------------------------------------------------------------------------
test ci
English | [简体中文](./README_cn.md)

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org.cn/documentation/docs/en/1.8/beginners_guide/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle, as the only independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms.
PaddlePaddle is originated from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.



## Installation

### Latest PaddlePaddle Release: [v1.8](https://github.com/PaddlePaddle/Paddle/tree/release/1.8)

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest features of PaddlePaddle.
### Install Latest Stable Release:
```
# Linux CPU
pip install paddlepaddle
# Linux GPU cuda10cudnn7
pip install paddlepaddle-gpu
# Linux GPU cuda9cudnn7
pip install paddlepaddle-gpu==1.8.5.post97

```
It is recommended to read [this doc](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/install/index_en.html) on our website.

Now our developers can acquire Tesla V100 online computing resources for free. If you create a program by AI Studio, you will obtain 12 hours to train models online per day. If you can insist on that for five consecutive days, then you will receive an extra 48 hours. [Click here to start](http://ai.baidu.com/support/news?action=detail&id=981).

## FOUR LEADING TECHNOLOGIES

- **Agile Framework for Industrial Development of Deep Neural Networks**

    The PaddlePaddle deep learning framework facilitates the development while lowering the technical burden, through leveraging a programmable scheme to architect the neural networks. It supports both declarative programming and imperative programming with both development flexibility and high runtime performance preserved.  The neural architectures could be automatically designed by algorithms with better performance than the ones designed by human experts.


-  **Support Ultra-Large-Scale Training of Deep Neural Networks**

    PaddlePaddle has made breakthroughs in ultra-large-scale deep neural networks training. It launched the world's first large-scale open-source training platform that supports the training of deep networks with 100 billions of features and trillions of parameters using data sources distributed over hundreds of nodes. PaddlePaddle overcomes the online deep learning challenges for ultra-large-scale deep learning models, and further achieved the real-time model updating with more than 1 trillion parameters.
     [Click here to learn more](https://github.com/PaddlePaddle/Fleet)


- **Accelerated High-Performance Inference over Ubiquitous Deployments**

    PaddlePaddle is not only compatible with other open-source frameworks for models training, but also works well on the ubiquitous developments, varying from platforms to devices. More specifically, PaddlePaddle accelerates the inference procedure with the fastest speed-up. Note that, a recent breakthrough of inference speed has been made by PaddlePaddle on Huawei's Kirin NPU, through the hardware/software co-optimization.
     [Click here to learn more](https://github.com/PaddlePaddle/Paddle-Lite)
     
     
- **Industry-Oriented Models and Libraries with Open Source Repositories**

     PaddlePaddle includes and maintains more than 100 mainstream models that have been practiced and polished for a long time in the industry. Some of these models have won major prizes from key international competitions. In the meanwhile, PaddlePaddle has further more than 200 pre-training models (some of them with source codes) to facilitate the rapid development of industrial applications.
     [Click here to learn more](https://github.com/PaddlePaddle/models)
     

## Documentation

We provide [English](http://www.paddlepaddle.org.cn/documentation/docs/en/1.8/beginners_guide/index_en.html) and
[Chinese](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/index_cn.html) documentation.

- [Basic Deep Learning Models](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html)

  You might want to start from how to implement deep learning basics with PaddlePaddle.


- [User Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/user_guides/index_en.html)

  You might have got the hang of Beginner’s Guide, and wish to model practical problems and build your original networks.
  
  
- [Advanced User Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/advanced_guide/index_en.html)

  So far you have already been familiar with Fluid. And the next step should be building a more efficient model or inventing your original Operator. 


- [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   Our new API enables much shorter programs.


- [How to Contribute](https://www.paddlepaddle.org.cn/documentation/docs/en/advanced_guide/addon_development/contribute_code/index_en.html)

   We appreciate your contributions!

## Communication

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 796771754 (PaddlePaddle).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
