# chargpt

chargpt 训练一个字符级的语言模型。

我们支持三种设置：1 种方便的设置和 2 种具有学术文献结果的“基准”设置：

- 用户指定的 `input.txt` 文件，我们将在该文件上训练 LM（例如，获取 tiny-shakespear（1.1MB 数据）[这里](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)）
- TODO [text8](http://mattmahoney.net/dc/textdata.html)：也源自维基百科文本，但所有 XML 都已删除，并且小写为仅包含 26 个英文字符和空格
- TODO [enwik8](http://prize.hutter1.net) 基准测试（“Hutter Prize”），维基百科 XML 转储的前 100MB 字节，包含 205 个唯一标记，包括英文字符和空格