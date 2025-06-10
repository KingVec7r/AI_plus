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
        您需要对图片中的内容进行识别。
        ### 输出格式 ### 
        您的输出由以下两部分组成，确保您的输出包含这两部分:
        ### 思考 ###
        考虑饮料外的标识，辨别饮料的种类，饮料容器。并且识别饮料为'有糖'或者'无糖'，给出你的思考过程。
        ### 识别结果 ### 
        若图中出现了饮料，请以json形式从左到右对他们进行描述，包括饮料：种类，是否有糖，饮料容器。
    """

    model_file = '/home/dancer/.cache/jr/workspace/aiplus/FM9G4B-V'
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

    image = Image.open('/home/dancer/.cache/jr/workspace/aiplus/cpm-9g-8b/FM9G4B-V/inference/step.jpg').convert('RGB')

    msgs = [[{'role': 'user', 'content': [image, prompt]}],
            [{'role': 'user', 'content': [image, prompt]}],
            [{'role': 'user', 'content': [image, prompt]}]]

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