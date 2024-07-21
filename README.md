
## minGPT 简介

![mingpt](mingpt.jpg)

minGPT 是一个使用 PyTorch 重新实现的 [GPT](https://github.com/openai/gpt-2) 模型，包含训练和推理部分。minGPT 旨在简洁、清晰、易于理解和学习，因为目前大多数可用的 GPT 模型实现都比较庞大。GPT 本身并不是一个复杂的模型，这个实现大约只有 300 行代码（参见 [mingpt/model.py](mingpt/model.py)）。它的核心原理是将一系列索引输入到 [Transformer](https://arxiv.org/abs/1706.03762) 中，然后输出序列中下一个索引的概率分布。大部分的复杂性在于为了提高效率，在批处理方面（包括跨样本和序列长度）进行了一些巧妙的设计。

**注意 (2023 年 1 月)**：虽然我可能会继续接受并修改一些细节，但 minGPT 处于半存档状态。有关最新进展，请参阅我的重写版本 [nanoGPT](https://github.com/karpathy/nanoGPT)。基本上，minGPT 被广泛引用（笔记本、博客、课程、书籍等），这使得我不太愿意进行更大的改动来推动代码向前发展。我还想稍微改变一下方向，从只关注教育到一些仍然简单易懂但有实用性的东西（重现中等规模的行业基准，接受一些权衡以提高运行时效率等）。

minGPT 库包含三个文件：[mingpt/model.py](mingpt/model.py) 包含实际的 Transformer 模型定义，[mingpt/bpe.py](mingpt/bpe.py) 包含一个稍微重构的字节对编码器，它可以像 OpenAI 在 GPT 中所做的那样在文本和整数序列之间进行转换，[mingpt/trainer.py](mingpt/trainer.py) 是（独立于 GPT 的）PyTorch 样板代码，用于训练模型。然后在 `projects` 文件夹中有一些使用该库的演示和项目：

- `projects/adder` 从头开始训练一个 GPT 来加数字（灵感来自 GPT-3 论文中的加法部分）
- `projects/chargpt` 在一些输入文本文件上训练一个 GPT 作为字符级语言模型
- `demo.ipynb` 以笔记本格式展示了在简单排序示例中 `GPT` 和 `Trainer` 的最小化使用
- `generate.ipynb` 展示了如何加载预训练的 GPT2 并根据一些提示生成文本

### 库安装

如果想在项目中 `import mingpt`：

```
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .
```

### 使用方法

以下是实例化 GPT-2（1.24 亿参数版本）的方法：

```python
from mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai 模型的词汇量
model_config.block_size = 1024  # openai 模型的块大小（即输入上下文长度）
model = GPT(model_config)
```

以下是训练模型的方法：

```python
# 你的 torch.utils.data.Dataset 子类，它生成长度不超过 1024 的示例
# torch LongTensor，其整数范围为 [0,50257)
train_dataset = YourDataset()

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # 许多可能的选项，请参阅文件
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
```

有关更具体的示例，请参阅 `demo.ipynb`。

### 单元测试

目前覆盖率还不算太高，但可以使用以下命令运行：

```
python -m unittest discover tests
```

### 待办事项

- 添加在任意给定文本文件上对 gpt-2 进行微调的演示
- 添加对话代理演示
- 更好地记录现有项目（adder、chargpt）的结果
- 添加混合精度和相关的训练扩展功能
- 分布式训练支持
- 在 projects/ 中重现一些基准测试，例如 text8 或其他语言建模
- 使用适当的日志记录而不是 print 语句
- 我可能应该有一个 requirements.txt 文件...
- 应该可以加载除 gpt2-\* 之外的许多其他模型权重

### 参考文献

代码：

- [openai/gpt-2](https://github.com/openai/gpt-2) 包含 TensorFlow 中的模型定义，但不包含训练代码
- [openai/image-gpt](https://github.com/openai/image-gpt) 的代码中有一些更现代的类似 gpt-3 的修改，也是很好的参考
- [huggingface/transformers](https://github.com/huggingface/transformers) 有一个 [语言建模示例](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)。它功能齐全，但因此也比较难以追踪。例如，一些大型函数在各种分支语句后面有多达 90% 的未使用代码，这些代码在简单的语言建模的默认设置中未使用

论文 + 一些实现说明：

#### 通过生成式预训练改进语言理解 (GPT-1)

- 我们的模型很大程度上遵循了原始的 Transformer 工作
- 我们训练了一个 12 层的仅解码器 Transformer，带有掩码自注意力头（768 维状态和 12 个注意力头）。对于位置感知前馈网络，我们使用了 3072 维的内部状态。
- Adam 最大学习率为 2.5e-4。（后来 GPT-3 对于这个模型大小使用了 6e-4）
- 学习率衰减：在前 2000 次更新中从零线性增加，并使用余弦调度退火至 0
- 我们在 64 个随机抽样的、连续的 512 个标记的最小批次上训练了 100 个 epoch
- 由于在整个模型中广泛使用了层归一化，因此 N(0, 0.02) 的简单权重初始化就足够了
- 字节对编码 (BPE) 词汇表，包含 40,000 次合并
- 残差、嵌入和注意力 dropout，速率为 0.1，用于正则化。
- (37) 中提出的 L2 正则化的修改版本，对所有非偏差或增益权重 w = 0.01
- 对于激活函数，我们使用了高斯误差线性单元 (GELU)。
- 我们使用了学习到的位置嵌入，而不是原始工作中提出的正弦版本
- 对于微调：我们向分类器添加了 dropout，速率为 0.1。学习率为 6.25e-5，批大小为 32。3 个 epoch。我们使用线性学习率衰减调度，在 0.2% 的训练中进行预热。λ 设置为 0.5。
- GPT-1 模型有 12 层，d_model 为 768，约 1.17 亿个参数 

##  语言模型是无监督的多任务学习者 (GPT-2)

-  层归一化被移动到每个子块的输入，类似于预激活残差网络
-  在最后一个自注意力块之后添加了一个额外的层归一化。
-  使用了一种改进的初始化方法，该方法考虑了残差路径上随模型深度累积的影响。我们在初始化时将残差层的权重缩放 1/√N，其中 N 是残差层的数量。(奇怪的是，在他们发布的代码中，我只能找到对旧的 0.02 的简单使用......在他们发布的 image-gpt 中，我发现它用于 c_proj，即使这样也只用于 attn，而不是 mlp。嗯。https://github.com/openai/image-gpt/blob/master/src/model.py)
-  词汇量扩大到 50,257
-  将上下文大小从 512 个标记增加到 1024 个标记
-  使用更大的批大小 512
-  GPT-2 使用了 48 层和 d_model 1600（而原始模型是 12 层和 d_model 768）。约 15.42 亿个参数

##  语言模型是少样本学习者 (GPT-3)

-  GPT-3：96 层，96 个头，d_model 为 12,288（1750 亿个参数）。
-  类似 GPT-1：12 层，12 个头，d_model 768（1.25 亿个参数）
-  我们使用与 GPT-2 相同的模型和架构，包括其中描述的改进的初始化、预归一化和可逆标记化
-  我们在 Transformer 的层中使用交替的密集和局部带状稀疏注意力模式，类似于稀疏 Transformer
-  我们始终将前馈层的大小设置为瓶颈层的四倍，dff = 4 ∗ dmodel
-  所有模型都使用 nctx = 2048 个标记的上下文窗口。
-  Adam，其中 β1 = 0.9，β2 = 0.95，eps = 10−8
-  所有模型都使用 0.1 的权重衰减来提供少量的正则化。（注意：我认为 GPT-1 使用的是 0.01，见上文）
-  将梯度的全局范数裁剪为 1.0
-  在前 3.75 亿个标记上进行线性学习率预热。然后使用余弦衰减将学习率降低到其值的 10%，持续 2600 亿个标记。
-  在训练的前 40-120 亿个标记（取决于模型大小）中，将批大小从较小的值（32k 个标记）线性增加到完整值。
-  始终使用完整的 2048 大小的上下文窗口，并使用特殊的文档结束标记分隔符

##  从像素生成式预训练 (Image GPT)

-  在处理图像时，我们选择恒等排列 πi = i，其中 1 ≤ i ≤ n，也称为光栅顺序。
-  我们通过使用 k 均值（k = 512）对 (R, G, B) 像素值进行聚类来创建我们自己的 9 位调色板。
-  我们最大的模型 iGPT-XL 包含 L = 60 层，并使用 d = 3072 的嵌入大小，总共 68 亿个参数。
-  我们的下一个最大模型 iGPT-L 本质上与 GPT-2 相同，具有 L = 48 层，但包含稍小的嵌入大小 d = 1536（而 GPT-2 为 1600），总共 14 亿个参数。
-  我们使用与 GPT-2 相同的模型代码，除了我们以与稀疏 Transformer (Child et al., 2019) 中相同的方式初始化权重，并将所有生成 logits 的投影初始化为零。
-  我们还训练了 iGPT-M，一个具有 L = 36 和 d = 1024 的 4.55 亿参数模型
-  iGPT-S，一个具有 L = 24 和 d = 512 的 7600 万参数模型（好的，有多少个头？看起来 Github 代码声称是 8 个）
-  在预训练 iGPT-XL 时，我们使用 64 的批大小并训练 200 万次迭代，对于所有其他模型，我们使用 128 的批大小并训练 100 万次迭代。
-  Adam，其中 β1 = 0.9，β2 = 0.95
-  学习率预热一个 epoch，然后衰减到 0
-  我们没有使用权重衰减，因为应用 0.01 的小权重衰减不会改变表示质量。
-  iGPT-S 学习率 0.003
-  未使用 Dropout。

### 许可证

MIT 
