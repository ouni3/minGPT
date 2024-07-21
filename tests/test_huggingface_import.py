"""
确保我们可以将 huggingface/transformer GPT 模型加载到 minGPT 中。
"""

import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

# -----------------------------------------------------------------------------

class TestHuggingFaceImport(unittest.TestCase):

    def test_gpt2(self):
        """
        测试 GPT2 模型。
        """
        model_type = 'gpt2' # 使用的 GPT 模型类型
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # 使用的设备
        prompt = "Hello!!!!!!!!!? 🤗, my dog is a little" # 测试 prompt

        # 创建 minGPT 和 huggingface/transformers 模型
        model = GPT.from_pretrained(model_type) # 从预训练模型创建 minGPT 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # 从预训练模型创建 huggingface/transformers 模型

        # 将模型移动到设备上
        model.to(device)
        model_hf.to(device)

        # 设置模型为评估模式
        model.eval()
        model_hf.eval()

        # 对输入 prompt 进行分词
        # ... 使用 minGPT 分词器
        tokenizer = BPETokenizer()
        x1 = tokenizer(prompt).to(device)
        # ... 使用 huggingface/transformers 分词器
        tokenizer_hf = GPT2Tokenizer.from_pretrained(model_type)
        model_hf.config.pad_token_id = model_hf.config.eos_token_id # 抑制警告
        encoded_input = tokenizer_hf(prompt, return_tensors='pt').to(device)
        x2 = encoded_input['input_ids']

        # 确保 logits 完全匹配
        logits1, loss = model(x1) # 获取 minGPT 模型的 logits
        logits2 = model_hf(x2).logits # 获取 huggingface/transformers 模型的 logits
        self.assertTrue(torch.allclose(logits1, logits2)) # 断言 logits 相等

        # 现在从每个模型中抽取 argmax 样本
        y1 = model.generate(x1, max_new_tokens=20, do_sample=False)[0] # 从 minGPT 模型中生成文本
        y2 = model_hf.generate(x2, max_new_tokens=20, do_sample=False)[0] # 从 huggingface/transformers 模型中生成文本
        self.assertTrue(torch.equal(y1, y2)) # 比较原始采样索引

        # 将索引转换为字符串
        out1 = tokenizer.decode(y1.cpu().squeeze()) # 将 minGPT 模型生成的文本解码为字符串
        out2 = tokenizer_hf.decode(y2.cpu().squeeze()) # 将 huggingface/transformers 模型生成的文本解码为字符串
        self.assertTrue(out1 == out2) # 比较输出字符串是否相等

if __name__ == '__main__':
    unittest.main()