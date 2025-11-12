#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from transformers import BertTokenizer

# import helpers.tokenization as tokenization
# import helpers.data_processing as dp


class BertCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, data_txt, bert_path, cache_file, batch_size, max_seq_length, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.position_ids_list = []
        
        # 读取输入数据并进行预处理
        with open(data_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i >= num_inputs:
                    break
                # 使用BERT分词器处理文本
                tokens = self.tokenizer.tokenize(line.strip())
                # 确保序列长度不超过最大长度
                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[:(max_seq_length - 2)]
                
                # 添加特殊标记
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                
                # 补齐或截断到最大序列长度
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                input_ids = input_ids[:max_seq_length]
                
                # 创建token_type_ids (0表示第一句话)
                token_type_ids = [0] * max_seq_length
                
                # 创建position_ids
                position_ids = list(range(max_seq_length))
                
                self.input_ids_list.append(input_ids)
                self.token_type_ids_list.append(token_type_ids)
                self.position_ids_list.append(position_ids)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        if num_inputs > len(self.input_ids_list):
            self.num_inputs = len(self.input_ids_list)
        else:
            self.num_inputs = num_inputs
        self.doc_stride = 128
        self.max_query_length = 64

        # Allocate enough memory for a whole batch.
        self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.int32.itemsize * self.batch_size) for binding in range(3)]

    def free(self):
        for dinput in self.device_inputs:
            dinput.free()

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        # 将输入数据从CPU复制到GPU
        batch_input_ids = []
        batch_token_type_ids = []
        batch_position_ids = []
        
        # 获取当前批次的数据
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx < self.num_inputs:
                batch_input_ids.extend(self.input_ids_list[idx])
                batch_token_type_ids.extend(self.token_type_ids_list[idx])
                batch_position_ids.extend(self.position_ids_list[idx])
        
        # 将数据复制到GPU设备内存
        cuda.memcpy_htod(self.device_inputs[0], np.array(batch_input_ids, dtype=np.int32))
        cuda.memcpy_htod(self.device_inputs[1], np.array(batch_token_type_ids, dtype=np.int32))
        cuda.memcpy_htod(self.device_inputs[2], np.array(batch_position_ids, dtype=np.int32))
        
        # 更新索引
        self.current_index += self.batch_size

        return self.device_inputs

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None

if __name__ == '__main__':
    data_txt = "calibrator_data.txt"
    bert_path = "bert-base-uncased"
    cache_file = "bert_calibrator.cache"
    batch_size = 1
    max_seq_length = 200
    num_inputs = 100
    cal = BertCalibrator(data_txt, bert_path, cache_file, batch_size, max_seq_length, num_inputs)

    cal.get_batch("input")
    cal.get_batch("input")
    cal.get_batch("input")