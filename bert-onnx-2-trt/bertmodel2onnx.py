
import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

print("pytorch:", torch.__version__)
# print("onnxruntime version:", ort.__version__)
# print("onnxruntime device:", ort.get_device())
print("transformers:", transformers.__version__)

BERT_PATH = 'bert-base-uncased'


def model_test(model, tokenizer, text):
    """测试BERT模型并保存输入输出数据用于后续验证"""
    print("==============model test===================")
    # 对输入文本进行编码，生成模型输入
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    # 定位mask token的位置索引
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)

    # 前向传播获取模型输出
    output = model(**encoded_input)
    print(output[0].shape)

    # 计算softmax概率并获取mask位置的预测结果
    logits = output.logits
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    # 获取top-10预测结果
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    # 解码并打印每个候选词的完整句子
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    # 保存输入和输出数据用于后续验证
    print("Saving inputs and output to case_data.npz ...")
    # 生成位置ID序列
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1)
    print(position_ids)
    # 转换输入数据为numpy数组
    input_ids=encoded_input['input_ids'].int().detach().numpy()
    token_type_ids=encoded_input['token_type_ids'].int().detach().numpy()
    print(input_ids.shape)

    # 保存数据到npz文件
    npz_file = BERT_PATH + '/case_data.npz'
    np.savez(npz_file,
             input_ids=input_ids,
             token_type_ids=token_type_ids,
             position_ids=position_ids,
             logits=output[0].detach().numpy())

    # 验证保存的数据
    data = np.load(npz_file)
    print(data['input_ids'])

def model2onnx(model, tokenizer, text):
    """
    将BERT模型转换为ONNX格式
    
    Args:
        model: 预训练的BERT模型实例
        tokenizer: BERT分词器，用于对输入文本进行编码
        text: 输入的文本字符串，用于构建模型输入和测试转换正确性
    """
    print("===================model2onnx=======================")
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    print(encoded_input)

    # 将模型设置为评估模式
    model.eval()
    export_model_path = BERT_PATH + "/model.onnx"
    opset_version = 12
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    
    # 使用PyTorch的ONNX导出功能将模型转换为ONNX格式
    torch.onnx.export(model,                                            # 正在运行的模型
                      args=tuple(encoded_input.values()),                      # 模型输入（单个输入或多个输入的元组）
                      f=export_model_path,                              # 保存模型的位置（可以是文件或类文件对象）
                      opset_version=opset_version,                      # 导出模型到ONNX的版本
                      do_constant_folding=False,                         # 是否执行常量折叠优化
                      input_names=['input_ids',                         # 模型的输入名称
                                   'attention_mask',
                                   'token_type_ids'],
                    output_names=['logits'],                    # 模型的输出名称
                    dynamic_axes={'input_ids': symbolic_names,        # 可变长度轴
                                  'attention_mask' : symbolic_names,
                                  'token_type_ids' : symbolic_names,
                                  'logits' : symbolic_names})
    print("Model exported at ", export_model_path)

if __name__ == '__main__':

    if not os.path.exists(BERT_PATH):
        print(f"Download {BERT_PATH} model first!")
        assert(0)

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertForMaskedLM.from_pretrained(BERT_PATH, return_dict = True)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."

    model_test(model, tokenizer, text)
    model2onnx(model, tokenizer, text)

