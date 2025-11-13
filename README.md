# infer-practical
深度学习模型推理优化实战案例

本项目提供了完整的深度学习模型TensorRT推理优化实践方案，涵盖了BERT、Vision Transformer (ViT) 和ERNIE三种主流模型的模型转换、量化优化和高性能推理部署。

## 🎯 项目概述

### 核心功能
- **模型转换**: 支持ONNX到TensorRT的高效转换
- **量化优化**: 提供FP16/INT8量化和校准功能
- **自定义算子**: 实现LayerNorm等自定义TensorRT插件

### 支持的模型架构

| 模型类型 | 描述 | 主要应用场景 |
|---------|------|-------------|
| **BERT** | Transformer-based语言模型 | 文本分类、情感分析、问答系统 |
| **Vision Transformer (ViT)** | 基于Transformer的视觉模型 | 图像分类、目标检测 |
| **ERNIE** | 百度中文语言模型 | 中文NLP任务、语义理解 |

## 📁 项目结构

```
infer-practical/
├── bert-onnx-2-trt/          # BERT模型TensorRT转换
│   ├── bertmodel2onnx.py     # BERT模型导出ONNX
│   ├── onnx2trt.py           # ONNX转TensorRT引擎
│   ├── builder.py            # TensorRT引擎构建器
│   ├── calibrator.py         # INT8量化校准器
│   ├── trt_helper.py         # TensorRT辅助工具
│   └── layernorm-plugin/     # LayerNorm自定义插件
├── vit2trt/                  # Vision Transformer转换
│   ├── model2onnx.py         # ViT模型导出ONNX
│   ├── trt_builder.py        # TensorRT构建器
│   ├── infer.py              # ViT推理引擎
│   ├── calibrator.py         # 量化校准
│   ├── trt_helper.py         # 辅助工具
│   ├── models/               # ViT模型定义
│   └── LayerNormPlugin/      # ViT专用LayerNorm插件
└── ernie2trt/                # ERNIE模型转换
    ├── ernie_model.py        # ERNIE模型定义
    ├── ernie_config.py       # ERNIE配置
    ├── infer.py              # ERNIE推理
    ├── API/                  # TensorRT API封装
    ├── ONNX/                 # ONNX相关工具
    └── infer_demo/           # C++推理示例
```

## 🚀 快速开始

### 环境要求

#### 硬件要求
- **GPU**: NVIDIA GPU (Compute Capability ≥ 6.0)
- **显存**: 建议8GB以上（根据模型大小）
- **内存**: 建议16GB以上

#### 软件依赖
```bash
# CUDA Toolkit
CUDA >= 11.0

# TensorRT
TensorRT >= 8.0

# Python依赖
pip install torch>=1.8.0
pip install onnx>=1.10.0
pip install numpy>=1.19.0
pip install transformers>=4.0.0
pip install opencv-python
```
### BERT模型推理优化

#### 1. 模型转换
```python
# 导出ONNX模型
python bertmodel2onnx.py \
    --model_name bert-base-uncased \
    --output_path bert_model.onnx

# 转换为TensorRT
python onnx2trt.py \
    --onnx_path bert_model.onnx \
    --precision fp16 \
    --workspace_size 1024
```

#### 2. 量化优化
```python
# INT8量化
python builder.py \
    --onnx_file bert_model.onnx \
    --precision int8 \
    --calibration_data calibration_data.npy \
    --output_engine bert_int8.trt
```

### Vision Transformer推理优化

#### 1. ViT模型导出
```python
python model2onnx.py \
    --model_type vit_base_patch16_224 \
    --output vit_base.onnx
```

#### 2. TensorRT推理
```python
python infer.py \
    --engine_file vit_base.trt \
    --input_image test_image.jpg
```

