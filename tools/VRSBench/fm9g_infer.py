"""
Use vllm python 10
"""

import time
import os
import json
import torch

from PIL import Image
from contextlib import contextmanager
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys

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
        # self.data = self.data[0:32]
     
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
            'ground_truth': item.get('ground_truth'),
            'type': item.get('type')  # 添加类型字段，默认为'unknown'    
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
    def __init__(self, data_path, model_file_path, infer_results_path, task="cap", batch_size=4):

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
        self.dataset = VRSBench(task, base_dir=data_path)

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
        question_type = []

        if local_rank == 0:
            print(f"Evaluating {len(self.dataset)} images from VRSBench for the task of {self.task}...")
        
        with timing(f"Total inference time", enable=(local_rank == 0)):
            # 分批处理数据
            # dist.barrier()
            for i,batch in enumerate(tqdm(
                dataloader, 
                desc=f"[rank{local_rank}]", 
                position = local_rank + 4,
                # disable=local_rank != 0
                )):  
                if i%15==0:
                    dist.barrier()              
                batch_images = [Image.open(path).convert('RGB') for path in batch['image_path']]
                
                # 过滤掉加载失败的图像
                valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
                if not valid_indices:
                    continue

                valid_images = [batch_images[i] for i in valid_indices]
                valid_gts = [batch['ground_truth'][i] for i in valid_indices]
                valid_ids = [batch['image_id'][i] for i in valid_indices]
                valid_questions = [batch['question'][i] for i in valid_indices]
                valid_types = [batch['type'][i] for i in valid_indices]
                    
                # 构建批处理输入
                if self.task == "cap" or self.task == "vqa":
                    prompt = valid_questions
 
                if self.task == "referring":
                    prompt = [f"""This is a remote sensing image.The object referred to by the given referring sentence is unique in the image. You need to provide the location of the referred object in the image in the form of a bounding box. The format of the bounding box is: <box> x1 y1 x2 y2</box>, where x1,y1 are the coordinates of the top-left corner of the bounding box, and x2,y2 are the coordinates of the bottom-right corner of the bounding box. Please note that the coordinates value of the bounding box are relative to the size of the image, ranging from 0 to 1000.\nEnsure that your response includes the coordinates of the bounding box and strictly follows the given format and should not contain any redundant content.\nresponse format: <box> x1 y1 x2 y2</box>.\nreferring sentence: '{q}'""" for q in valid_questions]

                if self.task == "vqa":
                    prompt = [f"""This is a remote sensing image. You need to answer the given question based on the image content, and your answer should be as concise as possible.\nquestion:'{q}'""" for q in valid_questions]

                batch_inputs = [[{'role': 'user', 'content': [image, prompt]}] for image, prompt in zip(valid_images,prompt)]
                
                # 模型推理
                if self.task == "referring":
                    max_new_tokens = 64
                else:
                    max_new_tokens = 1024

                if self.task == "cap":
                    sampling = True
                else:
                    sampling = False

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
                ground_truth.extend(valid_gts)
                image_ids.extend(valid_ids)
                questions.extend(valid_questions)
                question_type.extend(valid_types)
        
            # 收集所有进程的结果
            all_results = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_results, {
                'image_ids': image_ids,
                'question': questions,
                'ground_truth': ground_truth,
                'inference_results': inference_results,
                'type': question_type
            })

        dist.destroy_process_group()

        if local_rank == 0:
            merged_image_ids = []
            merged_question = []
            merged_ground_truth = []
            merged_inference_results = []
            merged_question_type = []
            
            for result in all_results:
                merged_image_ids.extend(result['image_ids'])
                merged_question.extend(result['question'])
                merged_ground_truth.extend(result['ground_truth'])
                merged_inference_results.extend(result['inference_results'])
                merged_question_type.extend(result['type'])
            # 保存结果
            infer_results = [
                {
                    "image_id": img_id,
                    "question": q,
                    "ground_truth": gt,
                    "inference_result": pred,
                    "type": q_type
                }
                for img_id, q, gt, pred, q_type in zip(merged_image_ids, merged_question,merged_ground_truth, merged_inference_results, merged_question_type)
            ]
            
            with open(self.results_file_path, "w", encoding="utf-8") as f:
                json.dump(infer_results, f, ensure_ascii=False, indent=2)  

            print(f"✅ 推理结果已保存至 {self.results_file_path}")

if __name__ == '__main__':
    # 模型文件路径
    model_file_path = './FM9G4B-V'

    # 推理结果保存路径
    infer_results_path = './tools/VRSBench/eval_result' 

    # VRSBenchs路径
    VRSBenchs_path = '/data/intelssd/jr/VRSBench'   

    # os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '7200'  # 设置NCCL调试信息和超时时间 2h
    
    # 启动DDP进程
    eval = VRSBenchEval(
        data_path = VRSBenchs_path,
        model_file_path = model_file_path, 
        infer_results_path = infer_results_path, 
        task = "referring", # 任务类型：cap, referring, vqa
        batch_size = 32  # cap_A100->64 referring_5880->32
        )

    eval.run()