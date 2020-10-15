
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org.cn/documentation/docs/en/1.8/beginners_guide/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

欢迎来到 PaddlePaddle GitHub

飞桨（PaddlePaddle）是目前国内唯一自主研发、开源开放、功能完备的产业级深度学习平台，集深度学习核心框架、基础模型库、端到端开发套件、工具组件和服务平台于一体。飞桨源于产业实践，致力于与产业深入融合，提供了领先的深度学习&机器学习任务开发、训练、部署能力，加速企业从算法研发到产业落地的过程。目前飞桨已广泛应用于工业、农业、服务业等，服务150多万开发者，与合作伙伴一起帮助越来越多的行业完成AI赋能。


## 安装
### PaddlePaddle最新版本: [v1.8](https://github.com/PaddlePaddle/Paddle/tree/release/1.8)

跟进PaddlePaddle最新特性请参考我们的[版本说明](https://github.com/PaddlePaddle/Paddle/releases)

### 安装最新稳定版本:
```
# Linux CPU
pip install paddlepaddle
# Linux GPU cuda10cudnn7
pip install paddlepaddle-gpu
# Linux GPU cuda9cudnn7
pip install paddlepaddle-gpu==1.8.5.post97

```
更多安装信息详见官网 [安装说明](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/install/index_cn.html)

PaddlePaddle用户可领取**免费Tesla V100在线算力资源**，训练模型更高效。**每日登陆即送12小时**，**连续五天运行再加送48小时**，[前往使用免费算力](https://ai.baidu.com/support/news?action=detail&id=981)。

## 四大领先技术

- **开发便捷的产业级深度学习框架**

    飞桨深度学习框架采用基于编程逻辑的组网范式，对于普通开发者而言更容易上手，符合他们的开发习惯。同时支持声明式和命令式编程，兼具开发的灵活性和高性能。网络结构自动设计，模型效果超越人类专家。
    

- **支持超大规模深度学习模型的训练**

    飞桨突破了超大规模深度学习模型训练技术，实现了支持千亿特征、万亿参数、数百节点的开源大规模训练平台，攻克了超大规模深度学习模型的在线学习难题，实现了万亿规模参数模型的实时更新。
    [查看详情](https://github.com/PaddlePaddle/Fleet)
    

- **多端多平台部署的高性能推理引擎**

    飞桨不仅兼容其他开源框架训练的模型，还可以轻松地部署到不同架构的平台设备上。同时，飞桨的推理速度也是全面领先的。尤其经过了跟华为麒麟NPU的软硬一体优化，使得飞桨在NPU上的推理速度进一步突破。
    [查看详情](https://github.com/PaddlePaddle/Paddle-Lite)


- **面向产业应用，开源开放覆盖多领域的工业级模型库。**

    飞桨官方支持100多个经过产业实践长期打磨的主流模型，其中包括在国际竞赛中夺得冠军的模型；同时开源开放200多个预训练模型，助力快速的产业应用。
    [查看详情](https://github.com/PaddlePaddle/models)


## 文档

我们提供 [英文](http://www.paddlepaddle.org.cn/documentation/docs/en/1.8/beginners_guide/index_en.html) 和
[中文](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/index_cn.html) 文档

- [深度学习基础教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/basics/index_cn.html)

   或许您想从深度学习基础开始学习飞桨
  

- [使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/user_guides/index_cn.html)

   或许您已经掌握了新手入门阶段的内容，期望可以针对实际问题建模、搭建自己网络
  

- [进阶使用](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/advanced_usage/index_cn.html)

   或许您已比较熟练使用PaddlePaddle来完成常规任务，期望获得更高效的模型或者定义自己的Operator
  
  
- [API Reference](http://paddlepaddle.org.cn/documentation/docs/zh/1.6/api_cn/index_cn.html)

   新的API支持代码更少更简洁的程序
   

- [贡献方式](http://paddlepaddle.org.cn/documentation/docs/zh/1.8/advanced_usage/development/contribute_to_paddle/index_cn.html)

   欢迎您的贡献!

## 交流与反馈

- 欢迎您通过[Github Issues](https://github.com/PaddlePaddle/Paddle/issues)来提交问题、报告与建议
- QQ群: 796771754 (PaddlePaddle)
- [论坛](http://ai.baidu.com/forum/topic/list/168): 欢迎大家在PaddlePaddle论坛分享在使用PaddlePaddle中遇到的问题和经验, 营造良好的论坛氛围

## 版权和许可证
PaddlePaddle由[Apache-2.0 license](LICENSE)提供
