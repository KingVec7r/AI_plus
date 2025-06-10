"""
Name chat.py
Date 2025/5/6 11:20
Version 1.0
TODO:
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    prompt = f"""### 背景 ###
        图片是遥感拍摄的居民区，你需要对图片中的内容进行识别。
        ### 输出格式 ### 
        您的输出由以下两部分组成，确保您的输出包含这两部分:
        ### 思考 ###
        考虑遥感图像的特点，分析图片中的目标数量，给出你的思考过程。
        ### 识别结果 ### 
        若图中出现了不同类型的目标，请以json形式对他们进行描述，包括 目标：目标种类，数量： 目标数量。
    """
    # prompt = f"""描述图片内容"""

    model_file = '/home/dancer/.cache/jr/workspace/aiplus/FM9G4B-V'
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

    image = Image.open('/home/dancer/.cache/jr/workspace/aiplus/cpm-9g-8b/FM9G4B-V/inference/2.png').convert('RGB')

    msgs = [{'role': 'user', 'content': [image, prompt]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    print("\n", "="*100, "\n")
    print(res)


    # 第二轮聊天，传递多轮对话的历史信息
    # msgs.append({"role": "assistant", "content": [res]})
    # msgs.append({"role": "user", "content": ["图中有几个箱子?"]})

    # answer = model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer
    # )
    # print("\n", "="*100, "\n")
    # print(answer)


    # ## 流式输出，设置：
    # # sampling=True
    # # stream=True
    # ## 返回一个生成器
    # msgs = [{'role': 'user', 'content': [image, prompt]}]
    # res = model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer,
    #     sampling=True,
    #     stream=True
    # )
    # print("\n", "="*100, "\n")
    # generated_text = ""
    # for new_text in res:
    #     generated_text += new_text
    #     print(new_text, flush=True, end='')