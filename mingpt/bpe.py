"""
BPE 是字节对编码器的缩写。它将任意的 UTF-8 字符串转换为整数序列，其中每个整数代表一小块经常出现的字符。

此实现基于 OpenAI 的 GPT-2 encoder.py：https://github.com/openai/gpt-2/blob/master/src/encoder.py

但略有修改，因为原始实现有点令人困惑。我还尝试添加尽可能多的注释，以解释我自己对代码的理解。
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    OpenAI 将每个可能的字节（实际上是整数 0..255）映射到一个直观表示它的 Unicode 字符。一些字节保留了它们的外观，
    因为它们不会造成任何麻烦。这些字节定义在列表 bs 中。例如：chr(33) 返回 "!"，所以在返回的字典中，我们简单地得到 d[33] -> "!"。
    然而，例如，chr(0) 是 '\x00'，这看起来很难看。所以 OpenAI 将这些字节映射到一个新的字符范围内，在这个范围内，chr() 
    返回一个美观的字符。所以在最终的字典中，我们得到 d[0] -> 'Ā'，它实际上是 chr(0 + 2**8)。
    特别地，空格字符是 32，我们可以通过 ord(' ') 看到这一点。这个函数会将空格 (32) 移动 256 位到 288，所以 d[32] -> 'Ġ'。
    所以这只是一个简单的字节 0..255 到 Unicode 字符的一对一映射，这些字符“看起来很漂亮”，要么是原始形式，
    要么是一个有趣的移位字符，比如 'Ā' 或 'Ġ' 等等。
    """
    # beautiful sequence,188个整数，这些整数以其原始形式呈现良好，不需要移位
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    
    # bs中的所有整数b都将在输出字典中简单地映射到chr(b)
    cs = bs[:] 
    
    # 现在获取需要移位的其他68个整数的表示形式
    # 每个都将映射到chr(256 + n)，其中n将在循环中从0...67增长
    n = 0
    for b in range(2**8):
        if b not in bs:# 如果此字节“难看”，则将其映射到下一个可用的“好看”字符        
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]

    #“不美观”的字符将被映射到扩展的 ASCII 码 (256 以上) 的字符。
    d = dict(zip(bs, cs))
    return d 

#返回word中包含所有相邻字符对的集合 pairs
def get_pairs(word):
    """
    返回一个由元组组成的集合，其中包含可迭代对象 word 中所有相邻元素组成的大字母组合。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        """

        **背景**

        * **预分词 (Pre-tokenization):** 在使用像 BPE 这样的子词单元分词算法之前，通常会先对文本进行预分词。预分词会将文本分割成更小的单元，例如单词、标点符号等。
        * **BPE 合并:** BPE 算法会迭代地将出现频率最高的字符对合并成一个新的子词单元。这个过程会持续进行，直到达到预设的词汇表大小或满足其他停止条件。
        * **问题:** 如果预分词阶段没有考虑到大小写敏感的情况，那么 BPE 算法可能会将同一个词的不同大小写形式视为不同的单元。例如，"The" 和 "the" 可能会被视为两个不同的词，即使它们在语义上是相同的。

        **注释解释**

        * **`re.IGNORECASE`:**  这是一个正则表达式标志，表示在进行正则表达式匹配时忽略大小写。
        * **作用:**  这段注释建议在预分词阶段使用 `re.IGNORECASE` 标志，以便将文本转换为小写形式。这样做可以确保 BPE 算法将同一个词的不同大小写形式视为相同的单元，从而提高分词的一致性和效率。

        **示例**

        假设我们有一个包含缩写词 "Mr." 的句子："Mr. Smith went to the store."

        * **不使用 `re.IGNORECASE`:** 预分词可能会将句子分割为 ["Mr.", "Smith", "went", "to", "the", "store", "."]。由于 "Mr." 中包含大写字母，BPE 算法可能会将其视为一个独立的单元，而不会将其与其他单词中的 "mr" 合并。
        * **使用 `re.IGNORECASE`:** 预分词会将句子转换为小写形式，并将其分割为 ["mr.", "smith", "went", "to", "the", "store", "."]。这样，BPE 算法就可以将 "mr." 与其他单词中的 "mr" 合并，从而生成更一致和有效的子词单元。

        **总结**

        在预分词阶段使用 `re.IGNORECASE` 标志可以有效地解决 BPE 分词在处理缩写词时的大小写敏感问题，提高分词质量。     

        """



        """
        这段正则表达式到底要查找什么？
        Python 正则表达式参考: https://docs.python.org/3/library/re.html
        竖线 | 表示“或”，因此 re.findall 会从左到右匹配文本，并将匹配的部分分割成块。
        \'s 会将类似 Andrej's 的内容拆分为 (Andrej, 's)。
        ?\p{L}+：可选的空格，后跟 1 个或多个 Unicode 字符，这些字符属于“字母”类别。
        ?\p{N}+：可选的空格，后跟 1 个或多个 Unicode 字符，这些字符属于“数字”类别。
        ?[^\s\p{L}\p{N}]+：可选的空格，然后是 1 个或多个既不是空格、字母也不是数字的字符。
        \s+(?!\S)：1 个或多个空白字符（例如空格、制表符等），除非它们后面跟着非空白字符。
        因此，这将匹配连续的空白字符序列，但排除该序列中的最后一个空白字符。 最后一个空白字符有机会匹配前面模式中的可选空格  ?。
        \s+：1 个或多个空白字符，可能用于捕获字符串末尾的完整尾随空白序列。
        简而言之：   
        我们对一些常见的撇号结构（'s、't、're 等）进行特殊处理，并将它们分成单独的标记。
        然后，我们将字符串分成连续的块：1) 字母、2) 数字、3) 非字母数字、4) 空白。
        总的来说，这个正则表达式的作用是将英文文本分割成单词、数字、标点符号等单元，并识别出一些常见的英文缩写。
        """
        # 字节编码器/解码器   d[32] -> 'Ġ'
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        
        # BPE（字节对编码）标记编码器/解码器  ('Ġ', 't')->[32]
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
       
        # BPE 合并列表，定义 BPE“树”，由要合并成标记 ab 的元组 (a,b) 组成
        #>>> bpe_merges = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        # >>> self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # >>> self.bpe_ranks
        # {('a', 'b'): 0, ('c', 'd'): 1, ('e', 'f'): 2}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 用于预分词的分割模式
        # 应该添加 re.IGNORECASE，以便 BPE 合并可以发生在缩写的首字母大写版本中 <-- 原始 openai 注释

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}




    def bpe(self, token):
        """
        此函数使用 self.bpe_ranks 将所有可能的 BPE（字节对编码）标记迭代地合并到树中。
        token 是一个字符串，表示单个“单词”（经过正则表达式分词后）以及字节编码后的结果，例如 “Ġthere”。
        #假设我们要处理另一个单词 token = "Ġgather"。
        # 处理流程：
        # 转换为字符元组：word = ('Ġ', 'g', 'a', 't', 'h', 'e', 'r')。
        # 获取所有相邻字符对：pairs = (('Ġ', 'g'), ('g', 'a'), ('a', 't'), ('t', 'h'), ('h', 'e'), ('e', 'r'))。
        # 进入迭代合并循环：
        # 查找 pairs 中排名最高的词对：('e', 'r')，合并得到 ('Ġ', 'g', 'a', 't', 'h', 'er')。
        # 再次查找排名最高的词对：('t', 'h')，合并得到 ('Ġ', 'g', 'a', 'th', 'er')。
        # 继续查找排名最高的词对：没有其他词对出现在 self.bpe_ranks 中，循环结束。
        # 最终结果为 "Ġga th er"。
        """
        # token 是单个“单词”的字符串，经过字节编码后，例如“Ġthere”。

        # 记忆化，提高效率
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)  # 将标记拆分为单个字符的元组，例如 ('Ġ', 't', 'h', 'e', 'r', 'e')
        pairs = get_pairs(word)  # 获取所有相邻字符对，例如 (('Ġ', 't'), ('t', 'h'), ('h', 'e'), ('e', 'r'), ('r', 'e'))

        if not pairs:
            return token  # 如果没有字符对，直接返回原始标记

        while True:
            # 找到可以合并的下一个靠前等级的字符对
            # 使用 lambda 表达式找到排名最靠前的字符对34
            bigram = min(pairs, key= lambda pair: self.bpe_ranks.get(pair, float('inf')))  
            if bigram not in self.bpe_ranks:
                break  # 如果没有更多字符对可以合并，则退出循环

            first, second = bigram  # 获取字符对的两个字符

            # 我们现在将在当前单词列表中将所有出现的 (first, second) 
            #替换为一个合并标记 first_second，并在输出列表 new_word 中
            new_word = []
            i = 0
            while i < len(word):
                # 在当前单词序列中查找下一个出现的 first
                try:
                    j = word.index(first, i)  # 从索引 i 开始查找字符 first 的位置
                    new_word.extend(word[i:j])  # 将从 i 到 j 的字符添加到 new_word
                    i = j  # 将 i 更新为 j
                except:
                    new_word.extend(word[i:])  # 如果找不到 first，则将剩余字符添加到 new_word
                    break  # 退出循环

                # 如果此 first 后面跟着 second，则将它们合并为一个
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 合并 first 和 second 并添加到 new_word
                    i += 2  # 将 i 向后移动两个位置
                else:
                    new_word.append(word[i])  # 将 first 添加到 new_word
                    i += 1  # 将 i 向后移动一个位置

            # 所有出现的 (first, second) 都已合并为 first_second
            new_word = tuple(new_word)  # 将 new_word 转换为元组
            word = new_word  # 将 word 更新为 new_word
            if len(word) == 1:
                break  # 如果 word 中只有一个元素，则退出循环
            else:
                pairs = get_pairs(word)  # 更新 pairs 列表

        # 将所有单词连接成一个字符串，并使用 ' ' 作为分隔符。请注意，
        # 到目前为止，所有字符都已进行字节编码，保证 ' ' 在实际数据中未使用，并且是一个“特殊”分隔符
        word = ' '.join(word)  # 使用空格连接所有字符

        # 缓存结果并返回
        self.cache[token] = word  # 将结果缓存到 self.cache 中
        return word  # 返回合并后的单词



    def encode(self, text):
        """ 
        输入字符串，输出整数列表（BPE 索引）

        # 预处理和分词:
        # 使用 self.pat 对输入文本进行分词，得到 tokens = ['This', 'is', 'a', 'test', '.']。
        # 字节编码和转换:

        # 遍历 tokens 列表，对每个 token 进行处理：
        # 例如，对于 token = 'This'：
        # 转换为字节序列： token_bytes = b'This'
        # 使用 self.byte_encoder 转换为 Unicode 字符串：token_translated = 'This'
        # 其他 token 也进行类似的转换。
        # BPE 合并:

        # 对每个 token_translated 应用 BPE 合并规则：
        # 例如，对于 token_translated = 'This'：
        # 由于 ('Th', 'is') 的优先级高于其他词对，因此将它们合并，得到 'This'。
        # 对于 token_translated = 'is'：
        # 由于 ('i', 's') 出现在 self.bpe_ranks 中，合并得到 'is'。
        # 其他 token_translated 也进行类似的处理。
        # 索引转换:

        # 将合并后的子词单元转换为词汇表索引：
        # ['This', 'is', 'a', 'test', '.'] -> [2, 3, 4, 5, 6]
        # 输出：

        # 最终输出 BPE 索引列表：[2, 3, 4, 5, 6]。
        """
        bpe_idx = []  # 初始化一个空的 BPE 索引列表

        # 将输入文本预先分词为字符串标记（粗略地说就是单词）
        tokens = re.findall(self.pat, text)  # 使用正则表达式 self.pat 对文本进行分词
       
        # 将每个标记处理成 BPE 整数
        for token in tokens:  # 遍历每个标记
            # 将标记编码为字节 (b'') 对象

            """
            text = "你好，世界！"  # 包含中文的字符串
            encoded_bytes = text.encode('utf-8') 
            print(encoded_bytes)  # 输出：b'\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81'
            """
            token_bytes = token.encode('utf-8')  # 使用 UTF-8 编码将字符串转换为字节
            # 将所有字节转换为其 Unicode 字符串表示形式并展平
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)  # 使用字节编码器将字节转换为 Unicode 字符
            # 根据 self.bpe_ranks 执行所有适用的 BPE 合并
            token_merged = self.bpe(token_translated).split(' ')  # 使用 BPE 算法对转换后的标记进行合并
            # 将所有 BPE 标记转换为整数
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]  # 使用编码器将 BPE 标记转换为整数索引
            # 扩展我们正在运行的所有输出整数列表
            bpe_idx.extend(token_ix)  # 将转换后的整数索引添加到 BPE 索引列表中
       
        return bpe_idx  # 返回 BPE 索引列表


    def encode_and_show_work(self, text):
        """ 
        调试函数，与 encode 相同，但返回所有中间结果 
        用于将文本编码为 BPE (Byte Pair Encoding) 索引序列，并返回所有中间结果，方便调试。
        """
        bpe_idx = []  # 最终的 BPE 索引列表
        parts = []  # 每个标记的中间结果列表
        tokens = re.findall(self.pat, text)  # 使用正则表达式对文本进行预分词
        for token in tokens:  # 遍历每个标记
            token_bytes = token.encode('utf-8')  # 将标记编码为字节
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)  # 将字节转换为 Unicode 字符
            token_merged = self.bpe(token_translated).split(' ')  # 对转换后的标记应用 BPE 合并
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]  # 将合并后的标记转换为 BPE 索引
            bpe_idx.extend(token_ix)  # 将 BPE 索引添加到最终列表中
            parts.append({  # 将中间结果添加到列表中
                'token': token,  # 原始标记
                'token_bytes': token_bytes,  # 字节表示
                'token_translated': token_translated,  # 转换后的标记
                'token_merged': token_merged,  # 合并后的标记
                'token_ix': token_ix,  # BPE 索引
            })
        out = {  # 返回结果字典
            'bpe_idx': bpe_idx,  # 最终的 BPE 索引序列
            'tokens': tokens,  # 预分词结果
            'parts': parts,  # 每个标记的中间结果
        }
        return out  # 返回结果字典

    def decode(self, bpe_idx):
        """ 输入整数列表，输出字符串 """
        # 对整数进行逆映射以获取标记
        tokens_merged = [self.decoder[token] for token in bpe_idx]  
        tokens_flat = ''.join(tokens_merged)

        # 反转字节编码器，例如将 'Ġ' 恢复为 ' '，并获取字节
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])

        # 恢复完整的 utf-8 字符串
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

def get_file(local_file, remote_file):
    """ 如果需要，将 remote_file 下载到 local_file """
    if not os.path.isfile(local_file):
        print(f"正在下载 {remote_file} 到 {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)

def get_encoder():
    """
    作用是加载预训练的 GPT BPE 编码器/解码器，并将其封装在一个 Encoder 对象中返回。
    该函数首先检查本地缓存中是否存在必要的文件，如果不存在则从远程服务器下载。
    然后，函数加载并解析这些文件，最终创建并返回一个 Encoder 对象。
    """
    home_dir = os.path.expanduser('~')  # 获取用户主目录
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')  # 缓存目录
    os.makedirs(cache_dir, exist_ok=True)  # 如果缓存目录不存在则创建

    # 加载 encoder.json，其中包含从标记到 BPE 索引的原始映射
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')  # 本地 encoder.json 文件路径
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'  # 远程 encoder.json 文件 URL
    get_file(encoder_local_file, encoder_remote_file)  # 下载文件（如果需要）
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)  # 加载 encoder.json 文件
    assert len(encoder) == 50257  # 断言：编码器大小应为 50257（256 个字节标记，50,000 个合并标记和 1 个特殊的 <|endoftext|> 标记）

    # 加载 vocab.bpe，其中包含 BPE 合并，即 BPE 树结构
    # 格式为元组 (a, b)，表示 (a, b) 将合并为一个标记 ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')  # 本地 vocab.bpe 文件路径
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'  # 远程 vocab.bpe 文件 URL
    get_file(vocab_local_file, vocab_remote_file)  # 下载文件（如果需要）
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()  # 读取 vocab.bpe 文件
    # 轻量级后处理：去除第一行的版本号，最后一行是空白行
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]  # 解析 BPE 合并
    assert len(bpe_merges) == 50000  # 断言：合并标记数量应为 50,000

    # 构造编码器对象并返回
    enc = Encoder(encoder, bpe_merges)  # 创建 Encoder 对象
    return enc  # 返回编码器对象

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ 
    BPETokenizer 类提供了一个方便的接口，用于在 PyTorch 环境中使用预训练的 BPE 编码器/解码器。
    它可以将文本编码为 PyTorch 张量，并将编码后的张量解码回文本。
    这对于将文本数据预处理后输入到 PyTorch 模型中非常有用。
    """

    def __init__(self):
        self.encoder = get_encoder()  # 初始化时获取一个Encoder实例

    def __call__(self, text, return_tensors='pt'):
        # 目前仅支持PyTorch；这里是为了匹配huggingface/transformers接口
        assert return_tensors == 'pt'  # 断言：确保返回类型是'pt'（PyTorch张量）
        # 目前仅支持单个字符串输入，将来可能支持字符串列表
        assert isinstance(text, str)  # 断言：确保输入是字符串类型
        # 编码并创建一个大小为1的"批次维度"
        idx = [self.encoder.encode(text)]  # 使用Encoder实例对文本进行编码
        # 封装成PyTorch张量
        out = torch.tensor(idx, dtype=torch.long)  # 创建一个PyTorch张量
        return out  # 返回PyTorch张量

    def decode(self, idx):
        # 确保现在是一个简单的1维张量
        assert idx.ndim == 1  # 断言：确保输入是一个1维张量
        # 将索引解码为文本
        text = self.encoder.decode(idx.tolist())  # 使用Encoder实例将索引解码为文本
        return text  # 返回解码后的文本




if __name__ == '__main__':

    # 这是一个编码示例
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()  # 获取一个编码器实例
    r = e.encode_and_show_work(text)  # 对文本进行编码并显示中间步骤

    print("原始文本是：")
    print(text)
    print("首先，文本会被预先分词，分解成块，结果是：")
    print(r['tokens'])  # 打印预分词后的标记列表
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("然后我们迭代每个块并依次处理它们...")
    for part in r['parts']:
        print(part)  # 打印每个块的详细信息，包括原始标记、字节表示、转换后的标记、合并后的标记和最终的BPE索引
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (请参考 Encoder.encode 中的代码，了解这些中间结果是什么)
    print("最终结果是连接并展平所有 token_ix：")
    print(r['bpe_idx'])  # 打印最终的BPE索引列表
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # 这将成为Transformer的整数输入序列
    print("准备馈送到Transformer！")
