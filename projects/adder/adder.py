"""
训练一个 GPT 模型来进行 n 位数的加法运算。
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # 系统配置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/adder'

    # 数据配置
    C.data = AdditionDataset.get_default_config()

    # 模型配置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # 训练器配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # 我们使用的模型非常小，因此学习率可以稍微高一些

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    创建 n 位数加法问题的数据集。例如，如果 n=2，则一个示例加法问题
    是 85 + 50 = 135。这个问题将被表示为 GPT 模型的以下字符串：

    "8550531"

    这是因为：
    - 我们丢弃了 + 和 =，因为它们不是必需的。我们只是将输入数字的各位数字连接在一起进行编码。
    - 结果 135 被反向编码，以便 GPT 模型更容易学习加法，因为加法算法的工作原理就是这样。

    再举一个例子，问题 6 + 39 = 45 将被编码为：

    "0639054"

    你会注意到，我们用零进行填充，以确保我们总是生成大小完全相同的字符串：n + n + (n + 1)。当 n=2 时，这是 7。
    在测试时，我们将通过提供前 2n 位数字来输入加法问题，并希望 GPT 模型能够正确地用接下来的 (n+1) 位数字完成序列。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2 # 加数和被加数的位数
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # 训练集/测试集

        # 将所有加法问题分成训练数据或测试数据
        ndigit = self.config.ndigit
        assert ndigit <= 3, "以下几行代码将非常占用内存，未来可能会重构以支持更大的位数"
        num = (10**ndigit)**2 # ndigit 位数的加法问题的总数
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 取整个数据集的 20%，或者最多 500 个
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # 数字 0..9 的词汇表大小

    def get_block_size(self):
        # a, b, a+b 的长度，加上 1 是为了潜在的进位溢出，
        # 但也减去 1，因为最后一位数字永远不会回传，
        # 因为没有明确的 <EOS> 标记需要预测，它是隐含的
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # 给定问题索引 idx，首先恢复相关的 a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # 计算加法问题 a + b 的“标签”
        c = a + b
        # 将 a、b、c 的数字编码为字符串
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # 反转 c 以使加法更容易
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # 将每个字符转换为其标记索引
        # x 将作为 GPT 的输入，y 将作为相关的预期输出
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # 预测序列中的下一个标记
        y[:ndigit*2-1] = -1 # 我们只会在输出位置进行训练。-1 将损失掩盖为零
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # 获取默认配置和命令行中的覆盖配置（如果有）
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # 构造训练集和测试集
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # 构造模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # 构造训练器对象
    trainer = Trainer(config.trainer, model, train_dataset)

    # 用于评估模型的辅助函数
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # 仅隔离输入序列的前两位数字
            d1d2 = x[:, :ndigit*2]
            # 让模型对序列的其余部分进行采样
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) # 使用贪婪的 argmax，而不是采样
            # 隔离采样序列的最后一位数字
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) # 将数字反转为其“正常”顺序
            # 从各个数字解码整数
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i # 手动计算真实值
            # 评估此批次中结果的正确性
            correct = (d3i_pred == d3i_gt).cpu() # 软件 1.0 与软件 2.0 的较量就在这行代码上，哈哈
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # 只打印最多 5 个错误，以便了解情况
                    mistakes_printed_already += 1
                    print("GPT 声称 %d + %d = %d，但真实值是 %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s 最终得分：%d/%d = %.2f%% 正确" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # 迭代回调函数
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"迭代时间 {trainer.iter_dt * 1000:.2f}ms; 迭代次数 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # 评估训练集和测试集的得分
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # 如果 ndigit=2，我们可以负担得起整个训练集，否则不行
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # 如果这是我们目前看到的最高分，则保存模型
            if score > top_score:
                top_score = score
                print(f"保存得分最高的模型，得分为 {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # 将模型恢复为训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # 运行优化
    trainer.run()