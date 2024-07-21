
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

"""
**1. `set_seed(seed)` 函数**

* **功能:**  这个函数用于设置 Python 内置的随机数生成器 (`random`)、NumPy 的随机数生成器 (`np.random`) 以及 PyTorch 的 CPU 和 GPU 随机数生成器的种子。
* **作用:**  在机器学习实验中，设置随机种子对于保证实验的可重复性至关重要。通过使用相同的种子，可以确保每次运行代码时生成相同的随机数序列，从而使实验结果可复现。

**2. `setup_logging(config)` 函数**

* **功能:**  这个函数用于设置实验的日志记录。
* **步骤:**
    * **创建工作目录:** 首先，它获取配置对象 `config` 中的 `work_dir` 属性，并创建该目录（如果不存在）。
    * **记录命令行参数:**  它将运行脚本时使用的所有命令行参数写入到工作目录下的 `args.txt` 文件中。
    * **记录配置:**  它将配置对象 `config` 转换为字典，并以 JSON 格式写入到工作目录下的 `config.json` 文件中。
* **作用:**  将命令行参数和配置信息保存到文件中，方便日后查看和复现实验。

**3. `CfgNode` 类**

* **功能:**  这是一个轻量级的配置类，用于存储和管理配置信息。
* **属性:**
    *  `__dict__`:  用于存储配置的键值对。
* **方法:**
    *  `__init__(self, **kwargs)`:  构造函数，使用传入的关键字参数初始化配置。
    *  `__str__(self)`:  返回配置的字符串表示，使用缩进来展示嵌套结构。
    *  `_str_helper(self, indent)`:  辅助函数，用于递归地生成缩进的字符串表示。
    *  `to_dict(self)`:  将配置对象转换为字典。
    *  `merge_from_dict(self, d)`:  从字典更新配置。
    *  `merge_from_args(self, args)`:  从命令行参数列表更新配置。
* **作用:**  `CfgNode` 类提供了一种结构化的方式来存储和管理配置信息，并提供了方便的方法来从字典和命令行参数更新配置。

**总结**

这些代码片段提供了一套用于管理配置、设置随机种子和记录日志的工具，这些工具在构建和运行机器学习实验时非常有用。 用。
"""


# -----------------------------------------------------------------------------
# 作用: 在机器学习实验中，设置随机种子对于保证实验的可重复性至关重要。通过使用相同的种子，
# 可以确保每次运行代码时生成相同的随机数序列，从而使实验结果可复现。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 将命令行参数和配置信息保存到文件中，方便日后查看和复现实验
def setup_logging(config):
    """ 单调乏味的簿记工作 """
    work_dir = config.system.work_dir  # 获取工作目录
    # 如果工作目录不存在，则创建
    os.makedirs(work_dir, exist_ok=True)
    # 记录参数（如果有）
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))  # 将所有参数写入 args.txt
    # 记录配置本身
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))  # 将配置以 JSON 格式写入 config.json



class CfgNode:
    """
    __init__(self, **kwargs): 构造函数，使用传入的关键字参数初始化配置。
    __str__(self): 返回配置的字符串表示，使用缩进来展示嵌套结构。
    _str_helper(self, indent): 辅助函数，用于递归地生成缩进的字符串表示。
    to_dict(self): 将配置对象转换为字典。
    merge_from_dict(self, d): 从字典更新配置。
    merge_from_args(self, args): 从命令行参数列表更新配置。
    作用: CfgNode 类提供了一种结构化的方式来存储和管理配置信息，并提供了方便的方法来从字典和命令行参数更新配置。
    """
    # TODO: 像 yacs 中那样转换为字典的子类？
    # TODO: 实现冻结以防止误操作？
    # TODO: 在读取/写入参数时进行额外的存在/覆盖检查？

    #使用 **kwargs 接收任意数量的关键字参数，并将这些参数更新到对象的 __dict__ 属性中。
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # 将生成对象字符串表示的逻辑委托给一个名为 _str_helper 的辅助方法
    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 需要一个辅助函数来支持嵌套缩进，以便进行漂亮的打印 """
        parts = []
        for k, v in self.__dict__.items():  # 遍历所有属性
            if isinstance(v, CfgNode):  # 如果属性值是另一个 CfgNode 实例
                parts.append("%s:\n" % k)  # 添加属性名和一个换行符
                parts.append(v._str_helper(indent + 1))  # 递归调用 _str_helper 以打印嵌套的 CfgNode
            else:
                parts.append("%s: %s\n" % (k, v))  # 添加属性名、属性值和一个换行符
        parts = [' ' * (indent * 4) + p for p in parts]  # 对每一行进行缩进
        return "".join(parts)  # 将所有行连接成一个字符串并返回

    def to_dict(self):
        """ 返回配置的字典表示 """
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}
        # 如果属性值是 CfgNode，则递归调用 to_dict()，否则直接返回值

    def merge_from_dict(self, d):
        """ 从字典更新配置 """
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        从命令行获取的字符串列表（即 sys.argv[1:]）更新配置。

        参数的预期格式为 `--arg=value`，并且 arg 可以使用 . 来表示嵌套的子属性。例如：

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:  # 遍历每个参数

            keyval = arg.split('=')  # 将参数按 '=' 分割
            assert len(keyval) == 2, "期望每个覆盖参数的格式为 --arg=value，获取到 %s" % arg
            key, val = keyval  # 解包键和值

            # 首先将值转换为 Python 对象
            try:
                val = literal_eval(val)
                """
                这里需要一些解释。
                - 如果 val 只是一个字符串，literal_eval 将抛出 ValueError
                - 如果 val 表示一个对象（例如 3、3.14、[1,2,3]、False、None 等），它将被创建
                """
            except ValueError:
                pass  # 如果无法转换，则保持为字符串

            # 找到要插入属性的相应对象
            assert key[:2] == '--'  # 确保键以 '--' 开头
            key = key[2:]  # 去除 '--'0

            keys = key.split('.')  # 按 '.' 分割键
            obj = self
            for k in keys[:-1]:  # 遍历除最后一个键之外的所有键
                obj = getattr(obj, k)  # 获取嵌套对象的属性
            leaf_key = keys[-1]  # 获取最后一个键（叶子键）

            # 确保此属性存在
            assert hasattr(obj, leaf_key), f"{key} 不是配置中存在的属性"

            # 覆盖属性
            print("命令行正在使用 %s 覆盖配置属性 %s" % (val, key))
            setattr(obj, leaf_key, val)  # 设置属性值
