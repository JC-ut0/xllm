# xLLM + NPU 技术博客系列

> 本系列博客旨在推广xLLM框架在NPU上的应用，扩大NPU生态影响力。

## 博客目录

| 序号 | 标题 | 文件 | 目标受众 | 预计字数 |
|------|------|------|----------|----------|
| 1 | xLLM：让国产芯片跑出GPU的性能 | [01_xllm_introduction.md](./01_xllm_introduction.md) | 通用 | ~3000 |
| 2 | xLLM架构揭秘：服务-引擎解耦的艺术 | [02_architecture_design.md](./02_architecture_design.md) | 架构师 | ~5000 |
| 3 | ACLGraph：NPU推理加速的秘密武器 | [03_aclgraph_optimization.md](./03_aclgraph_optimization.md) | 性能工程师 | ~4000 |
| 4 | xTensor：让NPU显存利用率提升30%的秘密 | [04_xtensor_memory.md](./04_xtensor_memory.md) | 系统工程师 | ~4000 |
| 5 | NPU能跑赢GPU吗？xLLM性能基准测试报告 | [05_performance_benchmark.md](./05_performance_benchmark.md) | 技术决策者 | ~3500 |
| 6 | 从零开始：Qwen3在NPU上的部署指南 | [06_deployment_guide.md](./06_deployment_guide.md) | 开发者 | ~3000 |
| 7 | DeepSeek-V3在NPU上的高效推理实践 | [07_moe_optimization.md](./07_moe_optimization.md) | 算法工程师 | ~4500 |
| 8 | xLLM在京东：日均亿级请求的推理实践 | [08_jd_production_practice.md](./08_jd_production_practice.md) | 技术管理者 | ~3000 |

## 内容概要

### 1. 开篇：国产芯片上的高性能LLM推理
- 背景与挑战：GPU供应紧张、国产芯片生态不完善
- xLLM核心特性：全面的模型支持、关键技术亮点
- 快速上手：环境准备、一键启动、性能初体验

### 2. 架构解析：服务-引擎解耦设计
- 传统推理框架的痛点
- xLLM架构设计：Service层、Engine层
- 核心调度策略：实例调度、请求调度
- PD分离架构：原理与实现

### 3. NPU专属优化：ACLGraph深度解析
- NPU推理的性能瓶颈
- ACLGraph技术原理
- xLLM中的实现：动态维度参数化、多shape复用显存池
- 性能测试：8%-10%吞吐提升

### 4. 显存管理：xTensor动态内存池
- LLM推理的显存挑战
- xTensor设计理念：物理页池预分配 + 虚拟地址连续映射
- 关键技术实现：页复用机制、异步预映射、算子适配
- 性能收益：内存利用率提升30%+

### 5. 性能基准：NPU vs GPU全面对比
- 测试环境与方法论
- 离线推理、在线服务、不同输入长度性能对比
- 深度分析：NPU优势场景、待优化方向
- 成本效益分析

### 6. 实战部署：Qwen3在NPU上的最佳实践
- 准备工作：硬件要求、软件环境
- 模型准备：下载、格式检查
- 部署实战：单卡、多卡、关键参数
- 性能调优：吞吐、延迟、显存优化
- 监控与运维

### 7. MoE优化：DeepSeek-V3高效推理
- MoE模型推理挑战：负载不均衡、通信开销、显存占用
- xLLM的MoE优化：GroupMatmul、EPLB、通信优化
- DeepSeek-V3实践：部署配置、性能数据
- 其他MoE模型支持

### 8. 生产落地：京东内部实践案例
- 业务背景：京东AI应用场景
- 技术方案：整体架构、关键组件
- 性能数据：日均亿级请求、成本收益
- 踩坑与解决：冷启动、长尾延迟、显存碎片

## 发布计划建议

| 博客 | 发布渠道 | 发布时间 | 配套资源 |
|------|----------|----------|----------|
| 1. 开篇 | 微信公众号、知乎、掘金 | 第1周 | GitHub Star引导 |
| 2. 架构解析 | 技术社区、公司博客 | 第2周 | 架构图素材 |
| 3. ACLGraph | 知乎、CSDN | 第3周 | 性能测试数据 |
| 4. xTensor | 技术社区 | 第4周 | 对比图表 |
| 5. 性能基准 | 全渠道 | 第5周 | 完整测试报告PDF |
| 6. 实战部署 | 掘金、CSDN | 第6周 | 部署脚本 |
| 7. MoE优化 | 知乎、技术社区 | 第7周 | 性能数据 |
| 8. 生产落地 | 微信公众号 | 第8周 | 案例白皮书 |

## 配套资源

### 官方链接
- **GitHub仓库**：https://github.com/jd-opensource/xllm
- **技术报告**：https://arxiv.org/abs/2510.14686
- **文档站点**：https://xllm.readthedocs.io/zh-cn/latest/
- **Docker镜像**：https://quay.io/repository/jd_xllm/xllm-ai

### 联系方式
- **GitHub Issues**：https://github.com/jd-opensource/xllm/issues
- **官方微信群**：请扫描README中的二维码

---

> **版权声明**：本系列博客遵循 Apache 2.0 许可证，欢迎转载，请注明出处。
