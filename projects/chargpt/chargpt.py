"""
训练一个字符级的语言模型。
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():
    """
    获取配置。
    """
    C = CN()

    # 系统配置
    C.system = CN()
    C.system.seed = 3407  # 随机种子
    C.system.work_dir = './out/chargpt'  # 工作目录

    # 数据配置
    C.data = CharDataset.get_default_config()

    # 模型配置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'  # 使用的GPT模型类型

    # 训练器配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # 模型很小，可以使用更大的学习率

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    字符数据集，用于生成字符批次。
    """

    @staticmethod
    def get_default_config():
        """
        获取默认配置。
        """
        C = CN()
        C.block_size = 128  # 文本块大小
        return C

    def __init__(self, config, data):
        """
        初始化字符数据集。
        """
        self.config = config

        chars = sorted(list(set(data)))  # 获取所有唯一字符
        data_size, vocab_size = len(data), len(chars)
        print('数据包含 %d 个字符， %d 个唯一字符。' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }  # 字符到索引的映射
        self.itos = { i:ch for i,ch in enumerate(chars) }  # 索引到字符的映射
        self.vocab_size = vocab_size  # 词汇表大小
        self.data = data  # 数据文本

    def get_vocab_size(self):
        """
        获取词汇表大小。
        """
        return self.vocab_size

    def get_block_size(self):
        """
        获取文本块大小。
        """
        return self.config.block_size

    def __len__(self):
        """
        获取数据集长度。
        """
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        """
        获取指定索引的数据项。
        """
        # 获取一个长度为 (block_size + 1) 的字符块
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # 将每个字符编码为整数
        dix = [self.stoi[s] for s in chunk]
        # 返回张量
        x = torch.tensor(dix[:-1], dtype=torch.long)  # 输入字符序列
        y = torch.tensor(dix[1:], dtype=torch.long)  # 目标字符序列
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # 获取默认配置和命令行参数
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # 构建训练数据集
    text = open('input.txt', 'r').read()  # 读取文本数据
    train_dataset = CharDataset(config.data, text)

    # 构建模型
    config.model.vocab_size = train_dataset.get_vocab_size()  # 设置词汇表大小
    config.model.block_size = train_dataset.get_block_size()  # 设置文本块大小
    model = GPT(config.model)

    # 构建训练器
    trainer = Trainer(config.trainer, model, train_dataset)

    # 批次结束时的回调函数
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            # 打印训练信息
            print(f"迭代时间 {trainer.iter_dt * 1000:.2f}ms; 迭代次数 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # 评估模型
            model.eval()
            with torch.no_grad():
                # 从模型中采样文本
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # 保存模型
            print("保存模型")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # 将模型恢复到训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # 运行训练
    trainer.run()
    