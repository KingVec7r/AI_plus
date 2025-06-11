"""
Name chat.py
Date 2025/5/6 11:20
Version 1.1 (Optimized for faster inference)
TODO: 增加模型量化支持和更细粒度的性能监控
"""

import time
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from contextlib import contextmanager
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from transformers import AutoModel, AutoTokenizer, TextStreamer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

import sys

# Suppress output context manager
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

@contextmanager
def timing(description: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    
    # 转换为时分秒格式
    hours = int(total_seconds // 3600)
    remaining_seconds = total_seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds = int(remaining_seconds % 60)
    milliseconds = int((remaining_seconds - seconds) * 1000)  # 取整到毫秒
    
    # 格式化输出（保留前导零，例如 01:02:03.456）
    time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    print(f"{description}: {time_format}")

class VRSBench(Dataset):
    def __init__(self, task="cap", base_dir="/data/jr/VRSBench"):
        json_path = {
            "cap": "VRSBench_EVAL_Cap.json",
            "referring": "VRSBench_EVAL_referring.json",
            "vqa": "VRSBench_EVAL_vqa.json"
        }
        eval_json_path = os.path.join(base_dir, json_path[task])
        self.task = task
        self.base_dir = base_dir
        self.img_base_dir = os.path.join(base_dir, "Images_val")
        self.data = self.load_json_file(eval_json_path)
        self.data = self.data[0:4]
     
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item.get('image_id')
        image_path = os.path.join(self.img_base_dir, image_id)

        return {
            'question_id': item.get('question_id'), 
            'image_id': image_id,
            'image_path': image_path,
            'question': item.get('question'),
            'ground_truth': item.get('ground_truth')
        }
    
    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 验证数据格式是否为列表
                if not isinstance(data, list):
                    raise ValueError("JSON文件内容不是列表格式")
                # 验证列表中的元素是否为字典
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError("列表中的元素不是字典格式")
                return data
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 不存在")
            return []
        except json.JSONDecodeError:
            print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
            return []
        except ValueError as ve:
            print(f"错误: {ve}")
            return []
        except Exception as e:
            print(f"发生未知错误: {e}")
            return []
        
class VRSBenchEval:
    def __init__(self, model_file_path, infer_results_path, task="cap", batch_size=4):

        parts = model_file_path.rstrip('/').split('/')
        model_name = parts[-1] if parts else ''

        model_config = {
            'trust_remote_code': True,
            'attn_implementation': 'sdpa',
            'torch_dtype': torch.bfloat16,
        }

        print(f"Loading model from {model_file_path}...")
        with suppress_output():
            self.model = AutoModel.from_pretrained(model_file_path, **model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_file_path, trust_remote_code=True)

        self.results_file_path = os.path.join(infer_results_path, f"{model_name}_VRSBench_{task}_eval_result.json")
        self.task = task
        self.batch_size = batch_size
        self.world_size = torch.cuda.device_count()
        self.dataset = VRSBench(task)

    def run(self):
        try:
            torch.multiprocessing.spawn(
                self.vrsbench_eval_ddp,
                # args=(),
                nprocs=self.world_size,
                # join=True
            )
        except Exception as e:
            print(f"evaluate failed: {e}")

    def vrsbench_eval_ddp(self, local_rank):

        if local_rank == 0:
            print("Start VRSBench evaluation with DDP...")

        # 初始化进程组
        torch.cuda.set_device(local_rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", world_size=self.world_size, init_method='env://')

        if local_rank == 0:
            print(f"Transfering model to DDP model...")
        
        model = self.model.cuda()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model.eval()

        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=local_rank, shuffle=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=2)

        image_ids = []
        ground_truth = []
        inference_results = []
        questions = []

        if local_rank == 0:
            print(f"Evaluating {len(self.dataset)} images from VRSBench for the task of {self.task}...")
        
        with timing(f"[rank{local_rank}] Total inference time"):
            # 分批处理数据
            for batch in tqdm(dataloader, disable=local_rank == 5):
                
                batch_images = [Image.open(path).convert('RGB') for path in batch['image_path']]
                
                # 过滤掉加载失败的图像
                valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
                if not valid_indices:
                    continue

                valid_images = [batch_images[i] for i in valid_indices]
                valid_gts = [batch['ground_truth'][i] for i in valid_indices]
                valid_ids = [batch['image_id'][i] for i in valid_indices]
                valid_questions = [batch['question'][i] for i in valid_indices]
                    
                # 构建批处理输入
                if self.task == "cap" or self.task == "vqa":
                    prompt = valid_questions

                if self.task == "referring":
                    prompt = [f"""This is a remote sensing image. You need to determine the object referred to by the given referring sentence. The object referred to by the referring sentence is unique in the image. You need to provide the location of the referred object in the image in the form of a bounding box. The format of the bounding box is: x1,y1,x2,y2, where x1,y1 are the coordinates of the top-left corner of the bounding box, and x2,y2 are the coordinates of the bottom-right corner of the bounding box. Please note that the coordinates of the bounding box are percentage values relative to the size of the image, ranging from 0 to 100.\nEnsure that your response includes the coordinates of the bounding box and strictly follows the following format: {{<x1><y1><x2><y2>}}.\nreferring sentence: '{q}'""" for q in valid_questions]

                batch_inputs = [[{'role': 'user', 'content': [image, prompt]}] for image, prompt in zip(valid_images,prompt)]
                
                # 模型推理
                with torch.no_grad():
                    res = model.module.chat(
                        image=None,
                        msgs=batch_inputs,
                        tokenizer=self.tokenizer,
                        sampling=False,
                        num_beams=1
                    )
                
                # 处理输出
                inference_results.extend(res)
                ground_truth.extend(valid_gts)
                image_ids.extend(valid_ids)
                questions.extend(valid_questions)
        
        # 收集所有进程的结果
        all_results = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_results, {
            'image_ids': image_ids,
            'question': questions,
            'ground_truth': ground_truth,
            'inference_results': inference_results
        })

        dist.destroy_process_group()

        if local_rank == 0:
            merged_image_ids = []
            merged_question = []
            merged_ground_truth = []
            merged_inference_results = []
            
            for result in all_results:
                merged_image_ids.extend(result['image_ids'])
                merged_question.extend(result['question'])
                merged_ground_truth.extend(result['ground_truth'])
                merged_inference_results.extend(result['inference_results'])

            # 保存结果
            infer_results = [
                {
                    "image_id": img_id,
                    "question": q,
                    "ground_truth": gt,
                    "inference_result": pred
                }
                for img_id, q, gt, pred in zip(merged_image_ids, merged_question,merged_ground_truth, merged_inference_results)
            ]
            
            with open(self.results_file_path, "w", encoding="utf-8") as f:
                json.dump(infer_results, f, ensure_ascii=False, indent=2)  

            print(f"✅ 推理结果已保存至 {self.results_file_path}")

            self.result_eval(infer_results)
        
    def result_eval(self, infer_results):    
        # 计算评估指标
        print("Calculating BLEU score...")
        bleu = BLEU()
        bleu_score = bleu.corpus_score([r['inference_result'] for r in infer_results], [[r['ground_truth'] for r in infer_results]])
        print(f"BLEU eval result: {bleu_score}")
        print(f"BLEU score: {bleu_score.score:.2f}")


if __name__ == '__main__':
    # 模型文件路径
    model_file_path = '/home/dancer/.cache/jr/workspace/aiplus/FM9G4B-V'

    # 推理结果保存路径
    infer_results_path = '/home/dancer/.cache/jr/workspace/aiplus/tools/VRSBench/eval_result'    

    # 任务类型：cap, referring, vqa
    task = "cap"
    
    batch_size = 8

    # os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '3600'  # 设置NCCL调试信息和超时时间
    
    # 启动DDP进程
    eval = VRSBenchEval(
        model_file_path = model_file_path, 
        infer_results_path = infer_results_path, 
        task = "cap", 
        batch_size = 4
        )

    eval.run()