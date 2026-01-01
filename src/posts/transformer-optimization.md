---
title: Optimizing Transformers for Production
date: Nov 15, 2023
category: Deep Learning
excerpt: Reducing inference latency by 3x using Quantization and ONNX Runtime.
---

Deploying Large Language Models (LLMs) in production is expensive. In this deep dive, we explore post-training quantization (PTQ) and how to use tools like ONNX Runtime to speed up inference significantly.

## Why Quantization?

Quantization reduces the precision of the numbers used to represent a model's parameters, which can lead to significant reductions in model size and inference latency...
