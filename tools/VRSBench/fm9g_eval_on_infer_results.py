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
from torchvision.ops import box_iou

import sys
import re

import logging

# 基本配置（一次性设置）
logging.basicConfig(
    level=logging.INFO,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    filename='./tools/VRSBench/eval_result/eval.log'  # 输出到文件（可选）
)
# 获取logger对象
logger = logging.getLogger(__name__)
# logger.debug('这是调试信息')
# logger.info('这是普通信息')
# logger.warning('这是警告信息')
# logger.error('这是错误信息')
# logger.critical('这是严重错误信息')



@contextmanager
def timing(description: str, enable: bool = True):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    
    # 转换为时分秒格式
    if enable:
        hours = int(total_seconds // 3600)
        remaining_seconds = total_seconds % 3600
        minutes = int(remaining_seconds // 60)
        seconds = int(remaining_seconds % 60)
        milliseconds = int((remaining_seconds - seconds) * 1000)  # 取整到毫秒
        
        # 格式化输出（保留前导零，例如 01:02:03.456）
        time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        print(f"\n{description}: {time_format}")
        
class VRSBenchEval:
    def __init__(self, data_path, model_file_path, infer_results_path, task="cap", batch_size=4):

        parts = model_file_path.rstrip('/').split('/')
        model_name = parts[-1] if parts else ''
        self.results_file_path = os.path.join(infer_results_path, f"{model_name}_VRSBench_{task}_eval_result.json")
        self.task = task

    
    def run(self):
        try:
            self.vrsbench_eval_ddp()
        except Exception as e:
            print(f"evaluate failed: {e}")

    def vrsbench_eval_ddp(self):
        infer_results = self.load_json_file(self.results_file_path)
        self.result_eval(infer_results)
        
    def result_eval(self, infer_results):    
        if self.task == "cap":
            # 计算BLEU分数
            print("Calculating BLEU score...")
            bleu = BLEU()
            bleu_score = bleu.corpus_score([r['inference_result'] for r in infer_results], [[r['ground_truth'] for r in infer_results]])
            print(f"BLEU eval result: {bleu_score}")
            print(f"BLEU score: {bleu_score.score:.2f}")

        elif self.task == "referring":
            # 计算IoU分数
            bbox_infer = [self.parse_bbox_infer(res) for res in [r['inference_result'] for r in infer_results]]
            bbox_gt = [self.parse_bbox_gt(gt) for gt in [r['ground_truth'] for r in infer_results]]

            boxes_gt_valid = []
            boxes_infer_valid = []
            for bbox_i, bbox_g in zip(bbox_infer, bbox_gt):
                if isinstance(bbox_i, list) and isinstance(bbox_g, list):
                    boxes_gt_valid.append(bbox_g)
                    boxes_infer_valid.append(bbox_i)
            boxes_a = torch.tensor(boxes_gt_valid) * 10  # GT框 0-100 -> 0-1000
            boxes_b = torch.tensor(boxes_infer_valid)
            iou = box_iou(boxes_a, boxes_b)
            iou = iou.diag().tolist()
            # 计算有效IoU率
            valid_iou_rate = len(iou) / len(bbox_infer)
            # 计算有效IoU率
            iou_avg = sum(iou) / len(iou)
            # 计算IoU@0.5
            iou_at_05 = sum(1 for i in iou if i > 0.5) / len(iou)
            # 计算IoU@0.7
            iou_at_07 = sum(1 for i in iou if i > 0.7) / len(iou) 
            # 输出结果
            print(f"有效IoU率: {valid_iou_rate:.2%}")
            print(f"平均IoU: {iou_avg:.2f}")
            print(f"Acc@0.5: {iou_at_05:.2%}")
            print(f"Acc@0.7: {iou_at_07:.2%}")

        elif self.task == "vqa": # vllm 大模型打分
            import ray
            from packaging.version import Version
            from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
            assert Version(ray.__version__) >= Version("2.44.1"), (
                "Ray version must be at least 2.44.1"
            )
            #   [r['type'] for r in infer_results]
            prompt = [f"""Question: {q}, Ground Truth Answer: {gt}, Predicted Answer: {pred}. Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., pond and swimming pool.""" for q, gt, pred in zip(
                [r['question'] for r in infer_results],
                [r['ground_truth'] for r in infer_results],
                [r['inference_result'] for r in infer_results]
            )]
            
            ds = ray.data.from_items([{"text": p} for p in prompt])
            print(ds.schema())

            size = ds.count()
            print(f"Size of dataset: {size} prompts")

            # Configure vLLM engine.
            config = vLLMEngineProcessorConfig(
                model_source="unsloth/Llama-3.1-8B-Instruct",
                engine_kwargs={
                    "enable_chunked_prefill": True,
                    "max_num_batched_tokens": 4096,
                    "max_model_len": 16384,
                },
                concurrency=1,  # set the number of parallel vLLM replicas
                batch_size=64,
            )

            # Create a Processor object, which will be used to
            # do batch inference on the dataset
            vllm_processor = build_llm_processor(
                config,
                preprocess=lambda row: dict(
                    messages=[
                        {"role": "system", "content": "You are a bot that responds with haikus."},
                        {"role": "user", "content": row["text"]},
                    ],
                    sampling_params=dict(
                        temperature=0.3,
                        max_tokens=250,
                    ),
                ),
                postprocess=lambda row: dict(
                    answer=row["generated_text"],
                    **row,  # This will return all the original columns in the dataset.
                ),
            )

            ds = vllm_processor(ds)

            # Peek first 10 results.
            # NOTE: This is for local testing and debugging. For production use case,
            # one should write full result out as shown below.
            outputs = ds.take(limit=10)

            for output in outputs:
                prompt = output["prompt"]
                generated_text = output["generated_text"]
                print(f"Prompt: {prompt!r}")
                print(f"Generated text: {generated_text!r}")
               
    def parse_bbox_infer(self, bbox_str):
        # 匹配模式: 数字序列，可能包含小数点
        pattern = r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>'
        match = re.search(pattern, bbox_str)
        try:
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                bbox =  [x1, y1, x2, y2]
                # 检查坐标是否在有效范围内
                if all(0 <= coord <= 1000 for coord in bbox):
                    return bbox
                else:
                    logger.info(f"Bounding box coordinates out of range (infer): {bbox_str}")
                    return 0
            else:
                logger.error(f"Error parsing bounding box (infer): {bbox_str}")
                return 1
        except Exception as e:
            logger.error(f"Unexpected error parsing bounding box (infer): {bbox_str}, Error: {e}")
            return 2
        
    def parse_bbox_gt(self, bbox_str):
        # 匹配模式: 数字序列，可能包含小数点
        pattern = r'<(\d+)><(\d+)><(\d+)><(\d+)>'
        match = re.search(pattern, bbox_str)
        try:
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                bbox =  [x1, y1, x2, y2]
                # 检查坐标是否在有效范围内
                if all(0 <= coord <= 100 for coord in bbox):
                    return bbox
                else:
                    logger.info(f"Bounding box coordinates out of range (gt): {bbox_str}")
                    return 0
            else:
                logger.error(f"Error parsing bounding box (gt): {bbox_str}")
                return 1
        except Exception as e:
            logger.error(f"Unexpected error parsing gt bounding box (gt): {bbox_str}, Error: {e}")
            return 2
        
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

if __name__ == '__main__':
    # 模型文件路径
    model_file_path = './FM9G4B-V'

    # 推理结果保存路径
    infer_results_path = './tools/VRSBench/eval_result' 

    # VRSBenchs路径
    VRSBenchs_path = '/data/jr/VRSBench'   

    # os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '7200'  # 设置NCCL调试信息和超时时间 2h
    
    # 启动DDP进程
    eval = VRSBenchEval(
        data_path = VRSBenchs_path,
        model_file_path = model_file_path, 
        infer_results_path = infer_results_path, 
        task = "vqa", # 任务类型：cap, referring, vqa
        batch_size = 32  # cap_A100->64 referring_5880->32
        )

    eval.run()