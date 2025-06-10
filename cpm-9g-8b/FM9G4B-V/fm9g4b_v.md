<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2:before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3:before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4:before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5:before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6:before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
</style>

# 九格多模态大模型使用文档

本文档介绍九格多模态大模型4B-V版本的训练、推理方式。

1. 使用原生huggingface transformer的generate函数进行推理
2. 使用DeepSpeed进行模型训练
3. 完成[模型下载](https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/FM9G4B-V.tar.gz)并依照步骤安装所需的各项依赖后，即可进行推理和训练。


## 目录

<!-- - [仓库目录结构](#仓库目录结构) -->

- [九格多模态大模型使用文档](#九格多模态大模型使用文档)
  - [目录](#目录)
  - [模型下载](#模型下载)
  - [环境配置](#环境配置)
  - [推理脚本示例](#推理脚本示例)
  - [训练脚本示例](#训练脚本示例)

## 模型下载

可在 [模型下载](https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/FM9G4B-V.tar.gz) 下载FM9G4B-V模型。

## 环境配置

完成模型下载后，需要安装所需的各项依赖。

### conda 环境安装

```shell
1. 使用python 3.10.16 创建conda环境
conda create -n fm9g4bv python=3.10.16

2. 激活环境
conda activate fm9g4bv

3. 使用 pip 安装 requirements.txt 中的依赖
#切换到FM9G4B-V目录下
cd FM9G4B-V
#安装依赖环境
pip install -r requirements.txt
```

## 推理脚本示例

### transformers原生代码推理脚本

此代码适用于4B-V模型单卡推理。在指定路径时，需指定pytorch_model.bin文件**所在目录**的路径，注意不是pytorch_model.bin文件本身的路径。

#### 推理代码：

```shell
#切换到FM9G4B-V/inference 目录下
cd FM9G4B-V/inference

#修改chat.py代码中model_file为pytorch_model.bin文件**所在目录**的路径

#运行推理示例代码
python chat.py
```

#### transformers原生代码推理脚本示例

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    prompt = f"""图片中出现了什么？"""

    model_file = 'xxxxx'
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

    image = Image.open('xxxxx.jpg').convert('RGB')

    msgs = [{'role': 'user', 'content': [image, prompt]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    print(res)
```

## 训练脚本示例

```shell
#切换到FM9G4B-V/finetune 目录下
cd FM9G4B-V/finetune

#修改finetune_ds.sh代码中以下参数，按照data.json格式准备训练数据
MODEL="pytorch_model.bin文件**所在目录**的路径"
DATA="FM9G4B-V/finetune/data.json"
EVAL_DATA="FM9G4B-V/finetune/data.json"   

#运行训练脚本
sh finetune_ds.sh
```
