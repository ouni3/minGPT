"""
简单的训练循环；适用于任何任意神经网络的样板代码，因此此文件中的内容实际上与 GPT 没有特别的关系。
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        """ 获取默认配置 """
        C = CN()  # 创建一个配置对象
        # 要训练的设备
        C.device = 'auto'  # 自动选择设备（例如：如果有GPU就使用GPU）
        # 数据加载器参数
        C.num_workers = 4  # 数据加载器使用的进程数
        # 优化器参数
        C.max_iters = None  # 最大迭代次数（None 表示没有限制）
        C.batch_size = 64  # 批次大小
        C.learning_rate = 3e-4  # 学习率
        C.betas = (0.9, 0.95)  # Adam 优化器的 beta 参数
        C.weight_decay = 0.1  # 权重衰减（仅应用于矩阵乘法权重）
        C.grad_norm_clip = 1.0  # 梯度范数裁剪阈值
        return C  # 返回配置对象

    def __init__(self, config, model, train_dataset):
        self.config = config  # 配置对象
        self.model = model  # 模型对象
        self.optimizer = None  # 优化器，稍后初始化
        self.train_dataset = train_dataset  # 训练数据集
        self.callbacks = defaultdict(list)  # 回调函数字典，用于在训练过程中执行自定义操作

        # 确定训练使用的设备
        if config.device == 'auto':  # 如果配置为自动选择设备
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有GPU则使用GPU，否则使用CPU
        else:
            self.device = config.device  # 使用配置中指定的设备
        self.model = self.model.to(self.device)  # 将模型移动到指定的设备
        print("running on device", self.device)  # 打印正在使用的设备

        # 以下变量将在稍后分配给训练器类，用于记录日志等
        self.iter_num = 0  # 当前迭代次数
        self.iter_time = 0.0  # 迭代耗时
        self.iter_dt = 0.0  # 每次迭代的时间间隔

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
            model, config = self.model, self.config  # 获取模型和配置对象

            # 设置优化器
            self.optimizer = model.configure_optimizers(config)  # 根据配置初始化优化器

            # 设置数据加载器
            train_loader = DataLoader(
                self.train_dataset,  # 训练数据集
                sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),  # 随机采样器，允许重复采样
                shuffle=False,  # 不进行 shuffle，因为已经使用了随机采样器
                pin_memory=True,  # 将数据固定在内存中，加速数据传输
                batch_size=config.batch_size,  # 批次大小
                num_workers=config.num_workers,  # 加载数据的进程数
            )

            model.train()  # 将模型设置为训练模式
            self.iter_num = 0  # 初始化迭代次数
            self.iter_time = time.time()  # 记录开始时间
            data_iter = iter(train_loader)  # 创建数据迭代器
            while True:  # 开始训练循环

                # 获取下一个批次数据 (x, y)，如果迭代器结束则重新初始化
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)  # 重新初始化数据迭代器
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]  # 将数据移动到指定的设备
                x, y = batch  # 分别获取输入数据和标签

                # 前向传播
                logits, self.loss = model(x, y)  # 计算模型输出和损失

                # 反向传播和参数更新
                model.zero_grad(set_to_none=True)  # 清空梯度
                self.loss.backward()  # 计算损失梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # 梯度裁剪，防止梯度爆炸
                self.optimizer.step()  # 更新模型参数

                self.trigger_callbacks('on_batch_end')  # 触发 batch 结束时的回调函数
                self.iter_num += 1  # 更新迭代次数
                tnow = time.time()  # 记录当前时间
                self.iter_dt = tnow - self.iter_time  # 计算迭代耗时
                self.iter_time = tnow  # 更新迭代时间

                # 终止条件
                if config.max_iters is not None and self.iter_num >= config.max_iters:  # 如果达到最大迭代次数则退出循环
                    break
