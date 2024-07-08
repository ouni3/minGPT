"""
GPT语言模型的完整定义，所有内容都在这个单个文件中。

参考资料：

OpenAI发布的官方GPT-2 TensorFlow实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
huggingface/transformers的PyTorch实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    实现了目前存在于Google BERT仓库中的GELU激活函数（与OpenAI GPT相同）。
    参考文献: Gaussian Error Linear Units (GELU) 论文: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        """
        实现了GELU激活函数的前向传播

        GELU函数定义如下：
        GELU(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        Args:
        x: 输入张量

        Returns:
        输出张量，经过GELU激活函数处理后的结果
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    一个普通的带有掩码的多头自注意力层，并在最后进行投影。
    这里可以使用 torch.nn.MultiheadAttention，但我在这里包含了一个显式实现，
    以表明这里没有什么可怕的。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有头的 key、query、value 投影，但在一个批次中
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码，确保注意力只应用于输入序列的左侧
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # 批量大小、序列长度、嵌入维度 (n_embd)

        # 计算批次中所有头的 query、key、value，并将 head 向前移动到批量维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有头输出并排重新组合

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ 一个朴素的 Transformer 块  """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT 语言模型 """

    @staticmethod
    def get_default_config():
        C = CN()  # 假设 CN 是一个配置管理类
        # 必须在配置中提供 model_type 或 (n_layer, n_head, n_embd)
        C.model_type = 'gpt'  # 模型类型
        C.n_layer = None  # Transformer 块数量
        C.n_head = None  # 注意力头的数量
        C.n_embd =  None  # 嵌入维度
        # 这些选项必须在外部填写
        C.vocab_size = None  # 词汇表大小
        C.block_size = None  # 输入序列长度
        # dropout 超参数
        C.embd_pdrop = 0.1  # 嵌入 dropout 概率
        C.resid_pdrop = 0.1  # 残差连接 dropout 概率
        C.attn_pdrop = 0.1  # 注意力 dropout 概率
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None  # 确保词汇表大小已提供
        assert config.block_size is not None  # 确保输入序列长度已提供
        self.block_size = config.block_size

        type_given = config.model_type is not None  # 是否提供模型类型
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])  # 是否提供所有参数
        assert type_given ^ params_given  # 确保仅提供模型类型或所有参数，但不同时提供

        if type_given:
            # 根据模型类型转换为详细配置
            config.merge_from_dict({
                # 名称遵循 huggingface 命名约定
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M 参数
                # GPT-2 配置
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M 参数
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M 参数
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 参数
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (还有很多...)
                # 我编造了这些微型模型
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        # Transformer 模型结构
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入层
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入层
            drop = nn.Dropout(config.embd_pdrop),  # dropout 层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer 块列表
            ln_f = nn.LayerNorm(config.n_embd),  # 最后的层归一化
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 用于预测下一个词的线性层

        # 初始化所有权重，并对残差连接应用特殊的缩放初始化，根据 GPT-2 论文
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量 (注意我们不计算 lm_head 中的解码器参数)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("参数数量: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        # 初始化权重
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 线性层权重初始化
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 线性层偏置初始化为零
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 嵌入层权重初始化
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)  # 层归一化偏置初始化为零
            torch.nn.init.ones_(module.weight)  # 层归一化权重初始化为一

       @classmethod
    def from_pretrained(cls, model_type):
        """
        通过复制 huggingface/transformers 检查点的权重来初始化预训练的 GPT 模型。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # 创建一个从头开始初始化的 minGPT 模型
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai 模型的词汇表大小
        config.block_size = 1024   # openai 模型的块大小
        model = GPT(config)
        sd = model.state_dict()

        # 初始化一个 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制权重，同时确保所有参数在名称和形状上都对齐且匹配
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]  # 忽略这些
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上 openai 检查点使用 "Conv1D" 模块，但我们只想使用一个普通的 nn.Linear。
        # 这意味着我们在导入这些权重时必须对其进行转置
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的 Conv1D 权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 将其他参数进行普通复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        这个长函数不幸做的事情非常简单，并且非常谨慎：
        我们将模型的所有参数分成两类：将进行权重衰减以进行正则化的参数和不会进行正则化的参数（偏差和层归一化/嵌入权重）。
        然后我们返回 PyTorch 优化器对象。
        """

        # 将所有参数分成将进行和不会进行正则化权重衰减的参数
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整的参数名称
                # 随机注意：因为 named_modules 和 named_parameters 是递归的
                # 我们会多次看到相同的张量 p。但是这样做
                # 允许我们知道任何张量 p 属于哪个父模块...
                if pn.endswith('bias'):
                    # 所有偏差都不会衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将进行权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重将不会进行权重衰减
                    no_decay.add(fpn)

        # 验证我们是否考虑了每个参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数 %s 同时出现在衰减/不衰减集合中！" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数 %s 未被划分到衰减/不衰减集合中！" \
                                                    % (str(param_dict.keys() - union_params), )

        # 创建 pytorch 优化器对象
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        """
        预测下一个词元并计算损失（如果提供了目标）。

        参数：
            idx：输入词元索引的张量，形状为 (batch_size, sequence_length)。
            targets：可选的目标词元索引的张量，形状与 idx 相同。

        返回值：
            一个元组 (logits, loss)，其中 logits 是预测的下一个词元的 logits，
            loss 是交叉熵损失（如果提供了目标），否则为 None。
        """
        device = idx.device  # 获取设备
        b, t = idx.size()  # 获取批量大小和序列长度
        assert t <= self.block_size, f"无法处理长度为 {t} 的序列，块大小仅为 {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # 创建位置索引

        # 前向传播 GPT 模型
        tok_emb = self.transformer.wte(idx)  # 词元嵌入，形状为 (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入，形状为 (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # 应用 dropout
        for block in self.transformer.h:
            x = block(x)  # 通过每个 Transformer 块
        x = self.transformer.ln_f(x)  # 应用最终的层归一化
        logits = self.lm_head(x)  # 预测下一个词元的 logits

        # 如果给定了目标，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        给定一个索引序列 idx (形状为 (b,t) 的 LongTensor)，完成
        序列 max_new_tokens 次，每次将预测结果反馈回模型。
        你很可能希望确保在此操作中处于 model.eval() 模式。

        参数：
            idx：输入词元索引的张量，形状为 (batch_size, sequence_length)。
            max_new_tokens：要生成的词元数量。
            temperature：控制生成文本随机性的温度参数。
            do_sample：是否从预测分布中采样词元，否则选择概率最高的词元。
            top_k：可选参数，用于限制采样到概率最高的 k 个词元。

        返回值：
            生成的词元索引的张量，形状为 (batch_size, sequence_length + max_new_tokens)。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，我们必须在 block_size 处截断它
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 前向传播模型以获取序列中索引的 logits
            logits, _ = self(idx_cond)
            # 获取最后一步的 logits 并按所需的温度进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选地将 logits 裁剪为仅包含概率最高的 k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 将 logits 转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样或选择概率最高的元素
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # 将采样的索引追加到正在运行的序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx