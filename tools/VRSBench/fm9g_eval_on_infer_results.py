import time
import os
import json
import torch

from contextlib import contextmanager
from tqdm import tqdm
from sacrebleu.metrics import BLEU

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
        
class VRSBenchEval:
    def __init__(self, data_path, model_file_path, infer_results_path, task=None, batch_size=4):
        parts = model_file_path.rstrip('/').split('/')
        model_name = parts[-1] if parts else ''

        self.llm_judge_results = []
        self.llm_type_results = []

        self.judge_model_path = "/data/intelssd/jr/weights/Qwen/Qwen2.5-7B-Instruct"

        self.results_file_path = os.path.join(infer_results_path, f"{model_name}_VRSBench_{task}_eval_result.json")
        self.results_file_dict = {t:os.path.join(infer_results_path, f"{model_name}_VRSBench_{t}_eval_result.json") for t in ['cap','referring','vqa']}

        self.task = task
        self.batch_size = batch_size
        self.world_size = torch.cuda.device_count()
        
        self.llm_judeg_results_file_path = os.path.join(infer_results_path, f"{model_name}_VRSBench_vqa_llm_judge_result.json")
          
    def run(self):
        self.vrsbench_eval_ddp()

    def vrsbench_eval_ddp(self):
        if self.task:
            infer_results = self.load_json_file(self.results_file_path)
            self.result_eval(infer_results)
        else:
            for t in ['cap','referring','vqa']:
                self.task = t
                self.results_file_path = self.results_file_dict[t]
                infer_results = self.load_json_file(self.results_file_path)
                self.result_eval(infer_results)
        
    def result_eval(self, infer_results, skip_llm=True):    
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
            # infer_results = infer_results[:32]

            type_list = ['Category', 'Presence', 'Quantity', 'Color', 'Shape', 'Size','Position','Direction','Scene','Reasoning']

            if not skip_llm:
                question_list = [r['question'] for r in infer_results]
                ground_truth_list = [r['ground_truth'] for r in infer_results]
                inference_result_list = [r['inference_result'] for r in infer_results]
                
                # 生成提示词
                prompts_for_judge = [f"""Question: {q}, Ground Truth Answer: {gt}, Predicted Answer: {pred}. Does the predicted answer match the ground truth? Answer with 1 for match and 0 for no match. ONLY output 0 or 1 —no analysis, explanations, or extra text. Synonyms (e.g., "pond" and "swimming pool") count as matches.""" for q, gt, pred in zip(
                    question_list, ground_truth_list, inference_result_list
                )]

                prompts_for_type = [f"""'Question: {q}, Answer: {gt}'. Select the most appropriate tag for the above QA pair. The tag should be chosen from the candidate list and should reflect the most prominent attribute or aspect that the Q&A focuses on. Your response should include only the tag word —no explanations, punctuation, or additional text. Candidate tags: {str(type_list)}""" for q, gt in zip(
                    question_list, ground_truth_list
                )]
                
                print('Generating llm judge results...')
                prompts = prompts_for_judge + prompts_for_type
                preds = self.llm_generate(prompts)

                llm_judges = preds[:len(preds)//2]
                llm_types = preds[len(preds)//2:]

                judeg_results = [[self.extract_first_number(lj) for lj in lj_i] for lj_i in llm_judges]
                type_results = [[self.extract_first_word(lt, type_list) for lt in lt_i] for lt_i in llm_types]

                final_judge_results = self.extract_majority(judeg_results)
                final_type_results = self.extract_majority(type_results)

                infer_results = [
                    {
                        **r,
                        # 'llm_judge':lj,
                        # 'llm_type':lt,
                        # 'llm_type_ori':lt_o,
                        'llm_judge_result':fj,
                        'llm_type_result':ft
                    } for r, lj, lt, lt_o, fj, ft in zip(
                        infer_results, 
                        judeg_results, 
                        type_results,
                        llm_types,
                        final_judge_results,
                        final_type_results
                    )
                ]
        
                # 保存结果
                with open(self.llm_judeg_results_file_path, 'w', encoding='utf-8') as f:
                    json.dump(infer_results, f, ensure_ascii=False, indent=4)
                print(f"✅ LLM 评估结果已保存至 {self.llm_judeg_results_file_path}")

            if skip_llm:
                # 已有llm评分结果
                infer_results = self.load_json_file(self.llm_judeg_results_file_path)

            count_dict = {k:[] for k in type_list}
            valid_judge = 0
            valid_category = 0
            vqa_right_count = 0
            for item in infer_results:
                if item.get('llm_judge_result') in ['0', '1']:
                    item['llm_judge_result'] = int(item['llm_judge_result'])
                    valid_judge += 1
                    vqa_right_count += item['llm_judge_result']

                if item.get('llm_type_result') in type_list:
                    count_dict[item.get('llm_type_result')].append(item.get('llm_judge_result'))
                    valid_category += 1

            eval_Indicator = {
                'LLM分类有效率': valid_category / len(infer_results),
                'LLM评分有效率': valid_judge / len(infer_results),
                'vqa总准确率': vqa_right_count / valid_judge,
                **{ t:sum(count_dict[t]) / len(count_dict[t]) for t in type_list}
            }

            for key, value in eval_Indicator.items(): 
                print(f"{key}: {value:.2%}")
           
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
        
    def llm_generate(self, prompts):
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(
            temperature=0.45, 
            top_p=0.9, 
            max_tokens=64,
            n=5
            )
        llm = LLM(
            model=self.judge_model_path,
            dtype='float16',
            # gpu_memory_utilization=0.5,
            tensor_parallel_size=torch.cuda.device_count())
        outputs = llm.generate(prompts, sampling_params)
        res_list = []
        for output in outputs:
            # n = 3
            all_responses = [o.text.strip() for o in output.outputs]
            res_list.append(all_responses)
            #  n = 1
            # response_text = output.outputs[0].text.strip()
            # res_list.append(response_text)        
        return res_list
    
    def extract_first_number(self, text):
        """提取字符串中的第一个数字"""
        pattern = r'\d+'  # 匹配一个或多个数字
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None

    def extract_first_word(self, text, type_list):
        """提取字符串中的第一个单词"""
        # pattern = r'(?:Tag:)?\s*([A-Z][a-z]*)'  # 匹配一个或多个字母组成的单词
        pattern = r'[A-Z][a-z]+'
        matches = re.findall(pattern, text)
        
        # 如果匹配到单词且第一个是"Tag"，则返回第二个
        if matches and matches[0] == "Tag" and len(matches) > 1:
            Tag = matches[1]
        elif matches:
            Tag = matches[0]
        if Tag in type_list:
            return Tag
        return None

    def extract_majority(self, pred_list):
        """从每个子列表中提取出现至少两次的元素，如果没有则返回None"""
        result = []
        
        for sublist in pred_list:
            # 统计每个元素出现的次数
            count_dict = {}
            for element in sublist:
                count_dict[element] = count_dict.get(element, 0) + 1
            
            # 查找出现至少两次的元素
            majority_element = None
            max_count = 0
            
            for element, count in count_dict.items():
                if count > max_count:
                    max_count = count
                    majority_element = element

            result.append(majority_element)
        
        return result

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
        # task = "vqa", # 任务类型：cap, referring, vqa
        batch_size = 32  # cap_A100->64 referring_5880->32
        )

    eval.run()