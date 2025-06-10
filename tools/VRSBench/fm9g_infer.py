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
     
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item.get('image_id')
        image_path = os.path.join(self.img_base_dir, image_id)
        return {
            'image_path': image_path,
            'ground_truth': item.get('ground_truth'),
            'image_id': item.get('image_id')
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
        
    
def vrsbench_eval_ddp(local_rank, world_size, model_file_path, infer_results_path, task="cap", batch_size=4):

    if local_rank == 0:
        print("Start VRSBench evaluation with DDP...")

    # 初始化进程组
    torch.cuda.set_device(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", world_size=world_size, init_method='env://')

    if local_rank == 0:
        print(f"Loading model from {model_file_path}...")
    
    # 模型加载
    model_config = {
        'trust_remote_code': True,
        'attn_implementation': 'sdpa',
        'torch_dtype': torch.bfloat16,
    }
    with suppress_output():
        model = AutoModel.from_pretrained(model_file_path, **model_config)
    model = model.cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()
    
    if local_rank == 0:
        print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_file_path, trust_remote_code=True)
    
    if local_rank == 0:
        print(f"Loading evaluate data...")
    
    # 创建分布式数据集和数据加载器
    if local_rank == 0:
        print(f"Building dataset...")
    
    dataset = VRSBench(task)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    image_ids = []
    ground_truth = []
    inference_results = []
    prompt = f"""Describe the image in detail"""

    if local_rank == 0:
        print(f"Evaluating {len(dataset)} images from VRSBench for the task of {task}...")
    
    with timing(f"[rank{local_rank}] Total inference time"):
        # 分批处理数据
        for batch in tqdm(dataloader, disable=local_rank != 0):
            
            batch_image_path = batch['image_path']
            batch_images = [Image.open(path).convert('RGB') for path in batch_image_path]
            batch_gts = batch['ground_truth']
            batch_ids = batch['image_id']
            
            # 过滤掉加载失败的图像
            valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
            if not valid_indices:
                continue

            valid_images = [batch_images[i] for i in valid_indices]
            valid_gts = [batch_gts[i] for i in valid_indices]
            valid_ids = [batch_ids[i] for i in valid_indices]
                
            # 构建批处理输入
            batch_inputs = [[{'role': 'user', 'content': [image, prompt]}] for image in valid_images]
            
            # 模型推理
            with torch.no_grad():
                res = model.module.chat(
                    image=None,
                    msgs=batch_inputs,
                    tokenizer=tokenizer,
                    sampling=False,
                    num_beams=1
                )
            
            # 处理输出
            inference_results.extend(res)
            ground_truth.extend(valid_gts)
            image_ids.extend(valid_ids)
    
    # 收集所有进程的结果
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, {
        'image_ids': image_ids,
        'ground_truth': ground_truth,
        'inference_results': inference_results
    })
    
    # 主进程合并结果并保存
    if local_rank == 0:
        merged_image_ids = []
        merged_ground_truth = []
        merged_inference_results = []
        
        for result in all_results:
            merged_image_ids.extend(result['image_ids'])
            merged_ground_truth.extend(result['ground_truth'])
            merged_inference_results.extend(result['inference_results'])
        
        # 保存结果
        infer_results = [
            {
                "image_id": img_id,
                "ground_truth": gt,
                "inference_result": pred
            }
            for img_id, gt, pred in zip(merged_image_ids, merged_ground_truth, merged_inference_results)
        ]
        
        with open(infer_results_path, "w", encoding="utf-8") as f:
            json.dump(infer_results, f, ensure_ascii=False, indent=2)  

        print(f"✅ 推理结果已保存至 {infer_results_path}")
        
        # 计算评估指标
        print("Calculating BLEU score...")
        bleu = BLEU()
        bleu_score = bleu.corpus_score([r['inference_result'] for r in infer_results], [[r['ground_truth'] for r in infer_results]])
        print(f"BLEU eval result: {bleu_score}")
        print(f"BLEU score: {bleu_score.score:.2f}")
    
    # 清理进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    # 模型文件路径
    model_file_path = '/home/dancer/.cache/jr/workspace/aiplus/FM9G4B-V'

    # 推理结果保存路径
    infer_results_path = '/home/dancer/.cache/jr/workspace/aiplus/tools/VRSBench/eval_result/fm9g_infer_results.json'    

    # 任务类型：cap, referring, vqa
    task = "cap"
    
    world_size = torch.cuda.device_count()
    batch_size = 8
    # world_size = 1
    print(f"使用 {world_size} 个GPU，DDP推理加速")
    
    # 启动DDP进程
    torch.multiprocessing.spawn(
        vrsbench_eval_ddp,
        # (local_rank, world_size, model_file_path, infer_results_path, task="cap", batch_size=4)
        args=(world_size, model_file_path, infer_results_path, task, batch_size),
        nprocs=world_size,
        # join=True
    )
