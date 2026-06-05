---
title: PD 分离 Connector 设计
date: 2026-06-04 21:40:00
updated: 2026-06-04 21:40:00
published: false
tags:
  - vLLM
  - Ascend
  - Context parallel
  - DP 负载均衡
  - KV Cache
  - 推理优化
categories:
  - 推理优化
---

> 项目 PR：[vllm-project/vllm-Ascend#4052](https://github.com/vllm-project/vllm-ascend/pull/4054) [vllm-project/vllm-Ascend#1659](https://github.com/vllm-project/vllm-ascend/pull/1659)