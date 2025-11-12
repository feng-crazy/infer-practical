#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

import pycuda.driver as cuda
import pycuda.autoinit

class TrtNetworkHelper():
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger):
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

    def set_layer_name(self, layer, name):
        """
        工具函数。设置trt层或插件的名称并打印输出形状。
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        工具函数。检查trt层。
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        工具函数。设置精度、设置层名称和检查trt层。
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        # 如果未提供输入名称，则自动生成一个名称
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        # 添加输入层到网络中
        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        # 标记张量为网络输出
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        # 创建常量层存储嵌入权重
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        # 使用gather操作根据索引获取对应的嵌入向量
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        if layer_name is None:
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)

        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        # GELU激活函数实现：GELU(x) = x * Φ(x) ，其中Φ(x)是标准正态分布的累积分布函数
        # 近似计算公式：GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def addLayerNorm(self, x, gamma, beta, layer_name=None, precision=None):
        """
        添加LayerNorm层
        :param x: 输入张量
        :param gamma: gamma参数
        :param beta: beta参数
        :param layer_name: 层名称
        :param precision: 精度
        :return: 输出张量
        """
        # 创建常量层存储gamma和beta参数
        gamma_constant = self.network.add_constant(gamma.shape, trt.Weights(gamma))
        beta_constant = self.network.add_constant(beta.shape, trt.Weights(beta))
        
        # 创建LayerNorm插件
        plugin = self.plugin_registry.get_plugin_creator("LayerNorm", "1", "")
        if plugin is None:
            raise RuntimeError("LayerNorm plugin not found")
            
        layer_norm_plugin = self.network.add_plugin_v2(
            [x, gamma_constant.get_output(0), beta_constant.get_output(0)], 
            plugin.create_plugin("LayerNorm", None)
        )

        if layer_name is None:
            layer_name = "nn.LayerNorm"
        else:
            layer_name = "nn.LayerNorm." + layer_name

        self.layer_post_process(layer_norm_plugin, layer_name, precision)

        return layer_norm_plugin.get_output(0)

    def addLinear(self, x, weight, bias, layer_name=None, precision=None):
        """
        添加全连接层(Linear)
        :param x: 输入张量
        :param weight: 权重矩阵
        :param bias: 偏置向量
        :param layer_name: 层名称
        :param precision: 精度
        :return: 输出张量
        """
        # 使用全连接层实现Linear操作
        weight_transposed = np.ascontiguousarray(np.transpose(weight))
        trt_weights = trt.Weights(weight_transposed)
        trt_bias = trt.Weights(bias)
        
        linear_layer = self.network.add_fully_connected(x, bias.size, trt_weights, trt_bias)
        
        if layer_name is None:
            layer_name = "nn.Linear"
        else:
            layer_name = "nn.Linear." + layer_name
            
        self.layer_post_process(linear_layer, layer_name, precision)
        
        return linear_layer.get_output(0)

    def addReLU(self, layer, x, layer_name=None, precision=None):
        # 添加ReLU激活函数层
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        """
        添加Softmax层
        :param x: 输入张量
        :param dim: 进行softmax的维度
        :param layer_name: 层名称
        :param precision: 精度
        :return: 输出张量
        """
        softmax_layer = self.network.add_softmax(x)
        # TensorRT中axis参数指定的是进行softmax的维度
        # 注意TensorRT中的维度索引可能与PyTorch不同
        softmax_layer.axes = 1 << (dim if dim >= 0 else len(x.shape) + dim)

        if layer_name is None:
            layer_name = "nn.Softmax"
        else:
            layer_name = "nn.Softmax." + layer_name

        self.layer_post_process(softmax_layer, layer_name, precision)

        return softmax_layer.get_output(0)

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        # 添加对数运算层
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    def addAdd(self, a, b, layer_name=None, precision=None):
        """
        添加两个张量的加法操作
        :param a: 第一个输入张量
        :param b: 第二个输入张量
        :param layer_name: 层名称
        :param precision: 精度
        :return: 输出张量
        """
        add_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        
        if layer_name is None:
            layer_name = "elementwise.add"
        else:
            layer_name = "elementwise.add." + layer_name
            
        self.layer_post_process(add_layer, layer_name, precision)
        
        return add_layer.get_output(0)

    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """
        添加缩放操作
        :param x: 输入张量
        :param scale: 缩放因子
        :param layer_name: 层名称
        :param precision: 精度
        :return: 输出张量
        """
        # 创建常量张量表示缩放因子
        scale_array = np.array([scale], dtype=np.float32)
        scale_const = self.network.add_constant((1,), trt.Weights(scale_array))
        
        # 使用元素级乘法实现缩放
        scale_layer = self.network.add_elementwise(x, scale_const.get_output(0), trt.ElementWiseOperation.PROD)
        
        if layer_name is None:
            layer_name = "tensor.scale"
        else:
            layer_name = "tensor.scale." + layer_name
            
        self.layer_post_process(scale_layer, layer_name, precision)
        
        return scale_layer.get_output(0)

    def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor:
        """
        添加矩阵乘法操作
        :param a: 第一个输入张量
        :param b: 第二个输入张量
        :param layer_name: 层名称
        :return: 输出张量
        """
        # 使用矩阵乘法层实现矩阵乘法
        matmul_layer = self.network.add_matrix_multiply(
            a, trt.MatrixOperation.NONE, 
            b, trt.MatrixOperation.NONE
        )
        
        if layer_name is None:
            layer_name = "tensor.matmul"
        else:
            layer_name = "tensor.matmul." + layer_name
            
        self.layer_post_process(matmul_layer, layer_name, None)
        
        return matmul_layer.get_output(0)


    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        # 添加常量张量到网络中
        trt_layer = self.network.add_constant(w.shape, w)

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """"""
        # 添加shuffle层用于转置和重塑张量
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x


class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list):
        # 获取输入数量
        nInput = len(inputs)

        bufferD = []
        # 分配GPU内存
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        # for i in range(0, self.engine.num_bindings):
            # print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

        # 初始化输出数组
        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        # 验证输出形状是否匹配
        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # 预热运行
        self.context.execute_v2(bufferD)

        # 记录推理时间
        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        # 将结果从GPU拷贝回CPU
        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        # 打印输出结果信息
        for i in range(0, len(outputs)):
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
            # print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)