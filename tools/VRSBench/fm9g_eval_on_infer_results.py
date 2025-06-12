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

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.data = prompts
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 直接返回原始文本
        return {"prompt": self.data[idx]}
        
class VRSBenchEval:
    def __init__(self, data_path, model_file_path, infer_results_path, task="cap", batch_size=4):
        parts = model_file_path.rstrip('/').split('/')
        model_name = parts[-1] if parts else ''

        self.llm_judge_results = []
        self.llm_type_results = []

        if task == "vqa":
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
        
        self.llm_judeg_results_file_path = os.path.join(infer_results_path, f"{model_name}_VRSBench_{task}_llm_judge_result.json")
        

    
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
            infer_results = infer_results[:32]

            type_list = ['Category', 'Presence', 'Quantity', 'Color', 'Shape', 'Size','Position','Direction','Scene','Reasoning']

            question_list = [r['question'] for r in infer_results]
            ground_truth_list = [r['ground_truth'] for r in infer_results]
            inference_result_list = [r['inference_result'] for r in infer_results]
            type_list = [r['type'] for r in infer_results]
            # 生成提示词

            prompts_for_judge = [f"""Question: {q}, Ground Truth Answer: {gt}, Predicted Answer: {pred}. Does the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., pond and swimming pool.""" for q, gt, pred in zip(
                question_list, ground_truth_list, inference_result_list
            )]

            prompts_for_type = [f"""Question: {q}, Answer: {gt}. Select the most appropriate tag (a single word) for the question based on the content of this Q&A pair. Your response should be chosen from the candidate tags provided and should not include any extra content. Candidate tags: {str(type_list)}""" for q, gt in zip(
                question_list, ground_truth_list
            )]
            
            print('Generating llm judge results...')

            try:
                torch.multiprocessing.spawn(
                    self.judge_and_get_type_ddp,
                    args=(prompts_for_judge, prompts_for_type),
                    nprocs=self.world_size,
                    # join=True
                )
            except Exception as e:
                print(f"evaluate failed: {e}")

            for i, (output_j, output_t) in enumerate(zip(self.llm_judge_results, self.llm_type_results)):
                infer_results[i]['judge_llm'] = output_j
                infer_results[i]['type_llm'] = output_t
            
            # 保存结果
            with open(self.llm_judeg_results_file_path, 'w', encoding='utf-8') as f:
                json.dump(infer_results, f, ensure_ascii=False, indent=4)
            print(f"✅ LLM 评估结果已保存至 {self.llm_judeg_results_file_path}")

    def judge_and_get_type_ddp(self, local_rank, prompts_j=None, prompts_t=None):

        if not prompts_j or not prompts_t:
            return None
        
        prompts_j_data = PromptDataset(prompts_j)
        prompts_t_data = PromptDataset(prompts_t)

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

        sampler_j = DistributedSampler(prompts_j_data, num_replicas=self.world_size, rank=local_rank, shuffle=False)
        sampler_t = DistributedSampler(prompts_t_data, num_replicas=self.world_size, rank=local_rank, shuffle=False)
        dataloader_j = DataLoader(prompts_j_data, batch_size=self.batch_size, sampler=sampler_j, num_workers=2)
        dataloader_t = DataLoader(prompts_t_data, batch_size=self.batch_size, sampler=sampler_t, num_workers=2)

        image_ids = []
        ground_truth = []
        inference_results = []
        questions = []
        question_type = []

        if local_rank == 0:
            print(f"\nEvaluating {len(prompts_j_data)} results for the task of judge...")
        
        with timing(f"\nTotal inference time for judge", enable=(local_rank == 0)):
            # 分批处理数据
            # dist.barrier()
            for i,batch in enumerate(tqdm(
                dataloader_j, 
                desc=f"[rank{local_rank}]", 
                position = local_rank + 4,
                # disable=local_rank != 0
                )):  
                if i%15==0:
                    dist.barrier()             
                    
                prompt = batch['prompt']

                batch_inputs = [[{'role': 'user', 'content': [p]}] for p in prompt]
                
                # 模型推理
                max_new_tokens = 64
                sampling = True

                with suppress_output():
                    with torch.no_grad():
                        res = model.module.chat(
                            image=None,
                            msgs=batch_inputs,
                            tokenizer=self.tokenizer,
                            sampling=sampling,
                            num_beams=1,
                            max_new_tokens = max_new_tokens,
                        )
                
                # 处理输出
                inference_results.extend(res)
        
            # 收集所有进程的结果
            all_results = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_results, {
                'inference_results': inference_results,
            })

        if local_rank == 0:        
            for result in all_results:
                self.llm_judge_results.extend(result['inference_results'])

            print(f"\nEvaluating {len(prompts_t_data)} results for the task of type...")
        
        with timing(f"\nTotal inference time for judge", enable=(local_rank == 0)):
            # 分批处理数据
            # dist.barrier()
            for i,batch in enumerate(tqdm(
                dataloader_t, 
                desc=f"[rank{local_rank}]", 
                position = local_rank + 4,
                # disable=local_rank != 0
                )):  
                if i%15==0:
                    dist.barrier()             
                    
                prompt = batch['prompt']

                batch_inputs = [[{'role': 'user', 'content': [p]}] for p in prompt]
                
                # 模型推理
                max_new_tokens = 64
                sampling = True

                with suppress_output():
                    with torch.no_grad():
                        res = model.module.chat(
                            image=None,
                            msgs=batch_inputs,
                            tokenizer=self.tokenizer,
                            sampling=sampling,
                            num_beams=1,
                            max_new_tokens = max_new_tokens,
                        )
                
                # 处理输出
                inference_results.extend(res)
        
            # 收集所有进程的结果
            all_results = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_results, {
                'inference_results': inference_results,
            })

        if local_rank == 0:        
            for result in all_results:
                self.llm_type_results.extend(result['inference_results'])

        dist.destroy_process_group()
                          
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
        batch_size = 16  # cap_A100->64 referring_5880->32
        )

    eval.run()