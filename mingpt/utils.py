
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    """ 一个轻量级的配置类，灵感来自 yacs """
    # TODO: 像 yacs 中那样转换为字典的子类？
    # TODO: 实现冻结以防止误操作？
    # TODO: 在读取/写入参数时进行额外的存在/覆盖检查？

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
            key = key[2:]  # 去除 '--'
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
