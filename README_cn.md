
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

欢迎来到 PaddlePaddle GitHub

飞桨(PaddlePaddle)以百度多年的深度学习技术研究和业务应用为基础，是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。目前，飞桨累计开发者265万，服务企业10万家，基于飞桨开源深度学习平台产生了34万个模型。飞桨助力开发者快速实现AI想法，快速上线AI业务。帮助越来越多的行业完成AI赋能，实现产业智能化升级。

## 安装

### PaddlePaddle最新版本: [v2.0](https://github.com/PaddlePaddle/Paddle/tree/release/2.0)

跟进PaddlePaddle最新特性请参考我们的[版本说明](https://github.com/PaddlePaddle/Paddle/releases)

### 安装最新稳定版本:
```
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```
更多安装信息详见官网 [安装说明](https://www.paddlepaddle.org.cn/install/quick)

PaddlePaddle用户可领取**免费Tesla V100在线算力资源**，训练模型更高效。**每日登陆即送10小时**，[前往使用免费算力](https://aistudio.baidu.com/aistudio/index)。

## 四大领先技术

- **开发便捷的产业级深度学习框架**

    飞桨深度学习框架采用基于编程逻辑的组网范式，对于普通开发者而言更容易上手，符合他们的开发习惯。同时支持声明式和命令式编程，兼具开发的灵活性和高性能。网络结构自动设计，模型效果超越人类专家。
    

- **支持超大规模深度学习模型的训练**

    飞桨突破了超大规模深度学习模型训练技术，实现了支持千亿特征、万亿参数、数百节点的开源大规模训练平台，攻克了超大规模深度学习模型的在线学习难题，实现了万亿规模参数模型的实时更新。
    [查看详情](https://github.com/PaddlePaddle/Fleet)
    

- **支持多端多平台的高性能推理部署工具**

    飞桨不仅广泛兼容第三方开源框架训练的模型部署，并且为不同的场景的生产环境提供了完备的推理引擎，包括适用于高性能服务器及云端推理的原生推理库 [Paddle Inference](https://paddle-inference.readthedocs.io/en/latest/product_introduction/summary.html)，面向分布式、流水线生产环境下自动上云、A/B测试等高阶功能的服务化推理框架 [Paddle Serving](https://github.com/PaddlePaddle/Serving)，针对于移动端、物联网场景的轻量化推理引擎 [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)，以及在浏览器、小程序等环境下使用的前端推理引擎 [Paddle.js](https://www.paddlepaddle.org.cn/paddle/paddlejs)。同时，透过与不同场景下的主流硬件高度适配优化及异构计算的支持, 飞桨的推理性能也领先绝大部分的主流实现。


- **面向产业应用，开源开放覆盖多领域的工业级模型库。**

    飞桨官方支持100多个经过产业实践长期打磨的主流模型，其中包括在国际竞赛中夺得冠军的模型；同时开源开放200多个预训练模型，助力快速的产业应用。
    [查看详情](https://github.com/PaddlePaddle/models)


## 文档

我们提供 [英文](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) 和
[中文](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html) 文档

- [使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)

   或许您想从深度学习基础开始学习飞桨
  
- [应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  
- [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)

   新的API支持代码更少更简洁的程序
   

- [贡献方式](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_contribution/index_cn.html)

   欢迎您的贡献!

## 交流与反馈

- 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/Paddle/issues)来提交问题、报告与建议
- QQ群: 793866180 (PaddlePaddle)
- [论坛](https://ai.baidu.com/forum/topic/list/168): 欢迎大家在PaddlePaddle论坛分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围

## 版权和许可证
PaddlePaddle由[Apache-2.0 license](LICENSE)提供
