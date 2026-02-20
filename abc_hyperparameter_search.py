#!/usr/bin/env python3
"""
ABC Vector 自动调参脚本
======================

功能：
- 对 ABC Vector 的核心参数进行网格搜索
- 自动记录日志和生成报告
- 错误时发送邮件通知
- 显示清晰的进度条
- 自动保存最佳参数的模型向量到 outputs/{dataset}_best/

使用方法：
    python abc_hyperparameter_search.py

作者：CoT Vectors Research
"""

import os
import sys
import json
import time
import shutil
import logging
import traceback
import smtplib
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itertools import product
from tqdm import tqdm
import gc

# ============================================================================
# 配置区 - 在此修改模型、数据集和路径
# ============================================================================

# 模型配置
MODEL_PATH = "/home/haichao/TA/ABCVector/models/Qwen2.5-Math-7B"
MODEL_NAME = "qwen"  # "qwen" 或 "llama"

# 数据集配置
DATASET = "math_easy"  # "gsm8k", "math_easy", "math_hard", "mmlu_pro"
DATA_PATH = "/home/haichao/TA/ABCVector/data"

# 输出路径
RESULTS_DIR = "./results"

# 最佳模型输出路径 (outputs/{dataset}_best/)
BEST_OUTPUT_BASE = "./outputs"

# 邮件配置 - 详细配置请修改 email_helper.py
# 收件人邮箱
EMAIL_RECIPIENT = "byboyuanzhang@gmail.com"

# ============================================================================
# 调参配置
# ============================================================================

# 参数搜索空间
PARAM_GRID = {
    "kl_beta": [0.5, 1.0, 2.0],
    "kl_warmup_steps": [0],
    "abc_learning_rate": [5e-5, 1e-4, 5e-4],
}

# 固定参数
FIXED_PARAMS = {
    "num_epochs": 10,
    "batch_size": 2,
    "gradient_accumulation_steps": 2,
    "abc_hidden_dim": 512,
    "sigma_min": 1e-4,
    "max_length": 1024,
    "warmup_ratio": 0.1,
    "weight_decay": 1e-3,
    "num_support_samples": 3000,
    "num_test_samples": 100,
    "max_new_tokens": 512,
    "num_beams": 3,
}

# 测试层范围
LAYERS = list(range(0, 27, 2))  # 0, 2, 4, ..., 26

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class LayerResult:
    """单层结果"""
    layer: int
    accuracy: float
    correct: int
    total: int
    gate: float = 0.0
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """单次实验结果"""
    params: Dict[str, Any]
    layer_results: List[LayerResult] = field(default_factory=list)
    avg_accuracy: float = 0.0
    max_accuracy: float = 0.0
    best_layer: int = -1
    total_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    def compute_stats(self):
        """计算统计信息"""
        valid_results = [r for r in self.layer_results if r.error is None]
        if valid_results:
            accuracies = [r.accuracy for r in valid_results]
            self.avg_accuracy = sum(accuracies) / len(accuracies)
            self.max_accuracy = max(accuracies)
            self.best_layer = valid_results[accuracies.index(self.max_accuracy)].layer


@dataclass
class SearchResults:
    """搜索结果汇总"""
    model_path: str
    model_name: str
    dataset: str
    start_time: str
    end_time: str = ""
    total_experiments: int = 0
    completed_experiments: int = 0
    failed_experiments: int = 0
    experiments: List[ExperimentResult] = field(default_factory=list)
    best_experiment_idx: int = -1
    baseline_accuracy: float = 0.0
    
    def find_best(self):
        """找到最佳实验"""
        valid_exps = [i for i, e in enumerate(self.experiments) 
                      if e.status == "completed"]
        if valid_exps:
            # 按 avg_accuracy 排序
            best_idx = max(valid_exps, key=lambda i: self.experiments[i].avg_accuracy)
            self.best_experiment_idx = best_idx


# ============================================================================
# 工具函数
# ============================================================================

def setup_logging(results_dir: str, dataset: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f"abc_tuning_{dataset}_{timestamp}.log")
    
    # 创建 logger
    logger = logging.getLogger("abc_tuning")
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 文件 handler - 详细日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台 handler - 简洁输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def send_email(subject: str, body: str, is_error: bool = False):
    """发送邮件通知（使用 email_helper 模块）"""
    try:
        # 尝试导入 email_helper 模块
        from email_helper import send_email as _send_email_impl
        _send_email_impl(subject, body, is_error, EMAIL_RECIPIENT)
    except ImportError:
        # 模块不存在，使用备用方案
        print(f"📧 [邮件通知] {subject}")
        if is_error:
            print(f"   (邮件模块未配置，请参考 email_helper.py)")
        # 尝试系统 mail 命令
        try:
            subprocess.run(
                ["mail", "-s", f"[ABC调参] {subject}", EMAIL_RECIPIENT],
                input=body.encode(),
                timeout=10,
                capture_output=True
            )
        except:
            pass
    except Exception as e:
        print(f"⚠️ 邮件发送失败: {e}")


def format_params(params: Dict[str, Any]) -> str:
    """格式化参数显示"""
    return ", ".join([f"{k}={v}" for k, v in params.items()])


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


# ============================================================================
# 核心调参逻辑
# ============================================================================

class ABCHyperparameterSearch:
    """ABC Vector 超参数搜索"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        dataset: str,
        data_path: str,
        results_dir: str,
        param_grid: Dict[str, List],
        fixed_params: Dict[str, Any],
        layers: List[int],
        best_output_base: str = "./outputs",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.dataset = dataset
        self.data_path = data_path
        self.results_dir = results_dir
        self.param_grid = param_grid
        self.fixed_params = fixed_params
        self.layers = layers
        
        # 创建输出目录
        self.output_dir = os.path.join(results_dir, dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 最佳模型输出目录: outputs/{dataset}_best/
        self.best_output_dir = os.path.join(best_output_base, f"{dataset}_best")
        
        # 当前全局最佳性能（用于判断是否需要更新）
        self.best_avg_accuracy = -1.0
        self.best_max_accuracy = -1.0
        self.best_params = None
        self.best_experiment_index = -1
        
        # 如果已有最佳目录，尝试加载之前的最佳性能
        self._load_existing_best()
        
        # 设置日志
        self.logger = setup_logging(self.output_dir, dataset)
        
        # 初始化结果
        self.search_results = SearchResults(
            model_path=model_path,
            model_name=model_name,
            dataset=dataset,
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        # 生成参数组合
        self.param_combinations = self._generate_param_combinations()
        self.search_results.total_experiments = len(self.param_combinations)
        
        # 模型和数据（延迟加载）
        self.model_wrapper = None
        self.tokenizer = None
        self.support_samples = None
        self.test_samples = None
    
    def _load_existing_best(self):
        """加载已有的最佳性能记录（用于恢复搜索）"""
        meta_path = os.path.join(self.best_output_dir, "best_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.best_avg_accuracy = meta.get("avg_accuracy", -1.0)
                self.best_max_accuracy = meta.get("max_accuracy", -1.0)
                self.best_params = meta.get("params", None)
                print(f"📂 已加载历史最佳记录: 平均准确率={self.best_avg_accuracy:.2f}%, "
                      f"最高准确率={self.best_max_accuracy:.2f}%")
            except Exception as e:
                print(f"⚠️ 无法加载历史最佳记录: {e}")
        
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            combinations.append(params)
        
        return combinations
    
    def _load_model_and_data(self):
        """加载模型和数据"""
        self.logger.info("=" * 70)
        self.logger.info("加载模型和数据...")
        self.logger.info("=" * 70)
        
        # 导入必要模块
        import torch
        from src.models import CoTModelWrapper, load_tokenizer
        from src.data_utils import load_dataset
        from src.utils import set_seed
        
        set_seed(42)
        
        # 加载模型
        self.logger.info(f"模型路径: {self.model_path}")
        self.model_wrapper = CoTModelWrapper(self.model_path, self.model_name)
        self.tokenizer = load_tokenizer(self.model_path)
        self.logger.info(f"模型加载完成: {self.model_wrapper.num_layers} 层, "
                        f"hidden_size={self.model_wrapper.hidden_size}")
        
        # 加载数据
        self.logger.info(f"数据集: {self.dataset}")
        self.logger.info(f"数据路径: {self.data_path}")
        
        self.support_samples = load_dataset(
            self.data_path, self.dataset, "train", 
            self.fixed_params["num_support_samples"]
        )
        self.test_samples = load_dataset(
            self.data_path, self.dataset, "test",
            self.fixed_params["num_test_samples"]
        )
        
        self.logger.info(f"支持集: {len(self.support_samples)} 样本")
        self.logger.info(f"测试集: {len(self.test_samples)} 样本")
    
    def _run_baseline(self) -> float:
        """运行基线评估"""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("基线评估 (无 CoT Vector 注入)")
        self.logger.info("=" * 70)
        
        from src.eval import run_baseline_evaluation
        
        baseline_results = run_baseline_evaluation(
            model_wrapper=self.model_wrapper,
            tokenizer=self.tokenizer,
            test_samples=self.test_samples,
            dataset_type=self.dataset,
            max_new_tokens=self.fixed_params["max_new_tokens"],
            num_beams=self.fixed_params["num_beams"],
            use_early_stopping=False,
        )
        
        accuracy = baseline_results["accuracy"]
        self.search_results.baseline_accuracy = accuracy
        self.logger.info(f"基线准确率: {accuracy:.2f}% "
                        f"({baseline_results['correct']}/{baseline_results['total']})")
        
        return accuracy
    
    def _save_best_checkpoints(
        self,
        layer_checkpoints: Dict[int, Dict[str, Any]],
        experiment_result: ExperimentResult,
        exp_idx: int,
    ):
        """
        保存最佳实验的所有层 checkpoint 到 outputs/{dataset}_best/
        
        目录结构:
            outputs/{dataset}_best/
            ├── best_meta.json          # 元信息（参数、性能、时间戳）
            ├── abc_L0.pt               # 各层 checkpoint
            ├── abc_L2.pt
            ├── abc_L4.pt
            └── ...
        
        Args:
            layer_checkpoints: {layer_idx: state_dict} 各层的模型状态
            experiment_result: 该实验的结果
            exp_idx: 实验编号
        """
        import torch
        
        self.logger.info("")
        self.logger.info("🏆 发现新的最佳性能！保存最佳模型...")
        self.logger.info(f"   旧最佳: 平均={self.best_avg_accuracy:.2f}%")
        self.logger.info(f"   新最佳: 平均={experiment_result.avg_accuracy:.2f}%, "
                        f"最高=L{experiment_result.best_layer} {experiment_result.max_accuracy:.2f}%")
        
        # 如果目录已存在，清空旧文件
        if os.path.exists(self.best_output_dir):
            shutil.rmtree(self.best_output_dir)
        os.makedirs(self.best_output_dir, exist_ok=True)
        
        # 保存每一层的 checkpoint
        saved_layers = []
        for layer_idx, state_dict in sorted(layer_checkpoints.items()):
            checkpoint_path = os.path.join(self.best_output_dir, f"abc_L{layer_idx}.pt")
            
            save_data = {
                **state_dict,
                "args": {
                    **experiment_result.params,
                    **self.fixed_params,
                    "model_path": self.model_path,
                    "model_name": self.model_name,
                    "dataset": self.dataset,
                    "layer_idx": layer_idx,
                },
            }
            torch.save(save_data, checkpoint_path)
            saved_layers.append(layer_idx)
        
        # 保存元信息
        meta = {
            "params": experiment_result.params,
            "fixed_params": self.fixed_params,
            "avg_accuracy": experiment_result.avg_accuracy,
            "max_accuracy": experiment_result.max_accuracy,
            "best_layer": experiment_result.best_layer,
            "baseline_accuracy": self.search_results.baseline_accuracy,
            "improvement_over_baseline": experiment_result.avg_accuracy - self.search_results.baseline_accuracy,
            "experiment_index": exp_idx,
            "total_time": experiment_result.total_time,
            "saved_layers": saved_layers,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "layer_details": [
                {
                    "layer": lr.layer,
                    "accuracy": lr.accuracy,
                    "correct": lr.correct,
                    "total": lr.total,
                    "gate": lr.gate,
                }
                for lr in experiment_result.layer_results
                if lr.error is None
            ],
        }
        
        meta_path = os.path.join(self.best_output_dir, "best_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 更新全局最佳记录
        self.best_avg_accuracy = experiment_result.avg_accuracy
        self.best_max_accuracy = experiment_result.max_accuracy
        self.best_params = experiment_result.params.copy()
        self.best_experiment_index = exp_idx
        
        self.logger.info(f"   已保存 {len(saved_layers)} 层 checkpoint 到: {self.best_output_dir}")
        self.logger.info(f"   元信息: {meta_path}")
        
        print(f"  🏆 最佳模型已更新 → {self.best_output_dir} "
              f"({len(saved_layers)} 层)")
    
    def run_search(self):
        """运行超参数搜索"""
        import torch
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("ABC Vector 超参数搜索")
        self.logger.info("=" * 70)
        self.logger.info(f"模型: {self.model_path.split('/')[-1]}")
        self.logger.info(f"数据集: {self.dataset}")
        self.logger.info(f"测试层: {self.layers}")
        self.logger.info(f"参数组合总数: {len(self.param_combinations)}")
        self.logger.info(f"最佳模型保存目录: {self.best_output_dir}")
        self.logger.info("")
        
        # 打印参数搜索空间
        self.logger.info("参数搜索空间:")
        for param, values in self.param_grid.items():
            self.logger.info(f"  {param}: {values}")
        self.logger.info("")
        
        self.logger.info("固定参数:")
        for param, value in self.fixed_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("=" * 70)
        
        if self.best_avg_accuracy > 0:
            self.logger.info(f"历史最佳: 平均={self.best_avg_accuracy:.2f}%, "
                            f"参数={format_params(self.best_params) if self.best_params else 'N/A'}")
        
        # 加载模型和数据
        self._load_model_and_data()
        
        # 运行基线
        baseline_acc = self._run_baseline()
        
        # 开始搜索
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("开始超参数搜索...")
        self.logger.info("=" * 70)
        
        # 主进度条
        total_combinations = len(self.param_combinations)
        
        for exp_idx, params in enumerate(self.param_combinations):
            exp_num = exp_idx + 1
            
            self.logger.info("")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"实验 {exp_num}/{total_combinations}")
            self.logger.info(f"参数: {format_params(params)}")
            self.logger.info(f"{'='*70}")
            
            try:
                # 使用进度条显示层级进度
                print(f"\n📊 实验 {exp_num}/{total_combinations}: {format_params(params)}")
                
                result = ExperimentResult(params=params.copy())
                result.status = "running"
                start_time = time.time()
                
                # 本次实验各层的 checkpoint（用于保存最佳模型）
                layer_checkpoints = {}
                
                # 层级进度条
                layer_pbar = tqdm(
                    self.layers, 
                    desc="  层测试",
                    ncols=100,
                    leave=True
                )
                
                for layer_idx in layer_pbar:
                    layer_pbar.set_description(f"  L{layer_idx:02d}")
                    
                    try:
                        # 创建并训练 ABC
                        abc_method = self._create_abc_method(layer_idx, params)
                        
                        # 训练（简化输出）
                        self._train_silent(abc_method)
                        
                        # 评估
                        eval_results = self._eval_silent(abc_method)
                        
                        layer_result = LayerResult(
                            layer=layer_idx,
                            accuracy=eval_results["accuracy"],
                            correct=eval_results["correct"],
                            total=eval_results["total"],
                            gate=abc_method.gate.item(),
                        )
                        
                        # 保存该层的 state_dict（内存中暂存）
                        layer_checkpoints[layer_idx] = abc_method.get_state_dict()
                        
                        # 更新进度条显示
                        layer_pbar.set_postfix({
                            "acc": f"{eval_results['accuracy']:.1f}%",
                            "gate": f"{abc_method.gate.item():.3f}"
                        })
                        
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        gc.collect()
                        error_msg = f"CUDA OOM at layer {layer_idx}"
                        self.logger.error(error_msg)
                        
                        # 发送错误邮件
                        send_email(
                            subject=f"❌ 错误: {self.dataset} 实验 {exp_num}",
                            body=f"实验参数: {format_params(params)}\n\n"
                                 f"错误信息: {error_msg}\n{str(e)}\n\n"
                                 f"已停止当前实验。",
                            is_error=True
                        )
                        
                        # 记录错误并跳过该实验
                        result.status = "failed"
                        result.error_message = error_msg
                        result.total_time = time.time() - start_time
                        self.search_results.experiments.append(result)
                        self.search_results.failed_experiments += 1
                        break
                        
                    except Exception as e:
                        layer_result = LayerResult(
                            layer=layer_idx,
                            accuracy=0.0,
                            correct=0,
                            total=len(self.test_samples),
                            error=str(e)[:200],
                        )
                        layer_pbar.set_postfix({"error": "⚠️"})
                    
                    result.layer_results.append(layer_result)
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                    gc.collect()
                
                layer_pbar.close()
                
                # 如果实验成功完成
                if result.status != "failed":
                    result.compute_stats()
                    result.status = "completed"
                    result.total_time = time.time() - start_time
                    self.search_results.experiments.append(result)
                    self.search_results.completed_experiments += 1
                    
                    # 打印结果摘要
                    diff = result.avg_accuracy - baseline_acc
                    print(f"  ✓ 完成: 平均={result.avg_accuracy:.2f}% (Δ{diff:+.2f}%), "
                          f"最佳=L{result.best_layer} {result.max_accuracy:.2f}%, "
                          f"耗时={format_time(result.total_time)}")
                    
                    self.logger.info(f"实验完成: 平均准确率={result.avg_accuracy:.2f}%, "
                                    f"最佳层={result.best_layer}, "
                                    f"最佳准确率={result.max_accuracy:.2f}%, "
                                    f"耗时={format_time(result.total_time)}")
                    
                    # ========== 检查是否为全局最佳，保存最佳模型 ==========
                    if result.avg_accuracy > self.best_avg_accuracy and layer_checkpoints:
                        self._save_best_checkpoints(
                            layer_checkpoints, result, exp_idx
                        )
                    else:
                        self.logger.info(f"  当前: {result.avg_accuracy:.2f}% "
                                        f"<= 最佳: {self.best_avg_accuracy:.2f}%, 不更新")
                
                # 释放本次实验的 checkpoint 内存
                del layer_checkpoints
                gc.collect()
                
            except Exception as e:
                error_msg = f"实验 {exp_num} 失败: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                
                # 发送错误邮件
                send_email(
                    subject=f"❌ 严重错误: {self.dataset} 实验 {exp_num}",
                    body=f"实验参数: {format_params(params)}\n\n"
                         f"错误信息: {error_msg}\n\n"
                         f"堆栈跟踪:\n{traceback.format_exc()}\n\n"
                         f"搜索已终止。",
                    is_error=True
                )
                
                # 保存当前进度
                self._save_intermediate_results()
                
                # 抛出异常终止
                raise RuntimeError(error_msg)
        
        # 搜索完成
        self.search_results.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.search_results.find_best()
        
        # 生成报告
        self._generate_report()
        
        # 发送完成邮件
        best_info = ""
        if self.best_avg_accuracy > 0 and self.best_params:
            best_info = (
                f"\n全局最佳配置:\n{format_params(self.best_params)}\n\n"
                f"全局最佳结果:\n"
                f"  平均准确率: {self.best_avg_accuracy:.2f}%\n"
                f"  最高准确率: {self.best_max_accuracy:.2f}%\n"
                f"  提升: {self.best_avg_accuracy - baseline_acc:+.2f}%\n\n"
                f"最佳模型已保存到: {self.best_output_dir}\n"
            )
        
        if self.search_results.best_experiment_idx >= 0:
            best_exp = self.search_results.experiments[self.search_results.best_experiment_idx]
            send_email(
                subject=f"✅ 完成: {self.dataset} 超参数搜索",
                body=f"超参数搜索已完成！\n\n"
                     f"数据集: {self.dataset}\n"
                     f"完成实验: {self.search_results.completed_experiments}/{self.search_results.total_experiments}\n"
                     f"基线准确率: {baseline_acc:.2f}%\n"
                     f"{best_info}\n"
                     f"详细报告已保存到: {self.output_dir}"
            )
    
    def _create_abc_method(self, layer_idx: int, params: Dict[str, Any]):
        """创建 ABC 方法实例"""
        from src.methods.abc_vector import ABCCoTVector
        
        return ABCCoTVector(
            model_wrapper=self.model_wrapper,
            tokenizer=self.tokenizer,
            layer_idx=layer_idx,
            dataset_type=self.dataset,
            abc_hidden_dim=self.fixed_params["abc_hidden_dim"],
            kl_beta=params["kl_beta"],
            kl_warmup_steps=params["kl_warmup_steps"],
            sigma_min=self.fixed_params["sigma_min"],
            learning_rate=params["abc_learning_rate"],
            weight_decay=self.fixed_params["weight_decay"],
            warmup_ratio=self.fixed_params["warmup_ratio"],
            num_epochs=self.fixed_params["num_epochs"],
            batch_size=self.fixed_params["batch_size"],
            gradient_accumulation_steps=self.fixed_params["gradient_accumulation_steps"],
            max_length=self.fixed_params["max_length"],
        )
    
    def _train_silent(self, abc_method):
        """静默训练（隐藏详细输出）"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            abc_method.train(self.support_samples, wandb_run=None)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _eval_silent(self, abc_method) -> Dict[str, Any]:
        """静默评估（隐藏详细输出）"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            results = abc_method.eval(
                test_samples=self.test_samples,
                max_new_tokens=self.fixed_params["max_new_tokens"],
                num_beams=self.fixed_params["num_beams"],
                use_early_stopping=False,
            )
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return results
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 JSON 结果
        results_file = os.path.join(
            self.output_dir, 
            f"abc_tuning_intermediate_{timestamp}.json"
        )
        
        # 转换为可序列化格式
        results_dict = {
            "model_path": self.search_results.model_path,
            "model_name": self.search_results.model_name,
            "dataset": self.search_results.dataset,
            "start_time": self.search_results.start_time,
            "baseline_accuracy": self.search_results.baseline_accuracy,
            "total_experiments": self.search_results.total_experiments,
            "completed_experiments": self.search_results.completed_experiments,
            "failed_experiments": self.search_results.failed_experiments,
            "best_avg_accuracy": self.best_avg_accuracy,
            "best_params": self.best_params,
            "best_output_dir": self.best_output_dir,
            "experiments": []
        }
        
        for exp in self.search_results.experiments:
            exp_dict = {
                "params": exp.params,
                "avg_accuracy": exp.avg_accuracy,
                "max_accuracy": exp.max_accuracy,
                "best_layer": exp.best_layer,
                "total_time": exp.total_time,
                "status": exp.status,
                "error_message": exp.error_message,
                "layer_results": [
                    {
                        "layer": lr.layer,
                        "accuracy": lr.accuracy,
                        "correct": lr.correct,
                        "total": lr.total,
                        "gate": lr.gate,
                        "error": lr.error,
                    }
                    for lr in exp.layer_results
                ]
            }
            results_dict["experiments"].append(exp_dict)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"中间结果已保存到: {results_file}")
    
    def _generate_report(self):
        """生成最终报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 JSON 结果
        json_file = os.path.join(
            self.output_dir, 
            f"abc_tuning_results_{timestamp}.json"
        )
        self._save_intermediate_results()  # 复用保存逻辑
        
        # 生成中文报告
        report_file = os.path.join(
            self.output_dir,
            f"abc_tuning_report_{timestamp}.md"
        )
        
        report_lines = [
            "# ABC Vector 超参数搜索报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 实验配置",
            "",
            f"- **模型**: {self.model_path.split('/')[-1]}",
            f"- **数据集**: {self.dataset}",
            f"- **测试层**: {self.layers}",
            f"- **开始时间**: {self.search_results.start_time}",
            f"- **结束时间**: {self.search_results.end_time}",
            f"- **最佳模型目录**: `{self.best_output_dir}`",
            "",
            "### 参数搜索空间",
            "",
            "| 参数 | 取值范围 |",
            "|------|----------|",
        ]
        
        for param, values in self.param_grid.items():
            report_lines.append(f"| {param} | {values} |")
        
        report_lines.extend([
            "",
            "### 固定参数",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
        ])
        
        for param, value in self.fixed_params.items():
            report_lines.append(f"| {param} | {value} |")
        
        report_lines.extend([
            "",
            "## 2. 实验结果总览",
            "",
            f"- **总实验数**: {self.search_results.total_experiments}",
            f"- **完成实验**: {self.search_results.completed_experiments}",
            f"- **失败实验**: {self.search_results.failed_experiments}",
            f"- **基线准确率**: {self.search_results.baseline_accuracy:.2f}%",
            "",
        ])
        
        # 最佳结果
        if self.best_avg_accuracy > 0 and self.best_params:
            improvement = self.best_avg_accuracy - self.search_results.baseline_accuracy
            
            report_lines.extend([
                "## 3. 🏆 最佳配置（已保存到磁盘）",
                "",
                f"**保存位置**: `{self.best_output_dir}`",
                "",
                "### 最佳参数",
                "",
                "| 参数 | 值 |",
                "|------|-----|",
            ])
            
            for param, value in self.best_params.items():
                report_lines.append(f"| {param} | {value} |")
            
            report_lines.extend([
                "",
                "### 最佳结果",
                "",
                f"- **平均准确率**: {self.best_avg_accuracy:.2f}%",
                f"- **最高准确率**: {self.best_max_accuracy:.2f}%",
                f"- **相比基线提升**: {improvement:+.2f}%",
                "",
            ])
            
            # 从 best_meta.json 加载各层详细结果
            meta_path = os.path.join(self.best_output_dir, "best_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    layer_details = meta.get("layer_details", [])
                    if layer_details:
                        report_lines.extend([
                            "### 各层详细结果",
                            "",
                            "| 层 | 准确率 | 正确/总数 | Gate值 | 相比基线 |",
                            "|-----|--------|-----------|--------|----------|",
                        ])
                        
                        for lr in sorted(layer_details, key=lambda x: x["accuracy"], reverse=True):
                            diff = lr["accuracy"] - self.search_results.baseline_accuracy
                            report_lines.append(
                                f"| L{lr['layer']} | {lr['accuracy']:.2f}% | "
                                f"{lr['correct']}/{lr['total']} | {lr['gate']:.4f} | {diff:+.2f}% |"
                            )
                except Exception:
                    pass
        elif self.search_results.best_experiment_idx >= 0:
            best_exp = self.search_results.experiments[self.search_results.best_experiment_idx]
            improvement = best_exp.avg_accuracy - self.search_results.baseline_accuracy
            
            report_lines.extend([
                "## 3. 🏆 最佳配置",
                "",
                "### 最佳参数",
                "",
                "| 参数 | 值 |",
                "|------|-----|",
            ])
            
            for param, value in best_exp.params.items():
                report_lines.append(f"| {param} | {value} |")
            
            report_lines.extend([
                "",
                "### 最佳结果",
                "",
                f"- **平均准确率**: {best_exp.avg_accuracy:.2f}%",
                f"- **最高准确率**: {best_exp.max_accuracy:.2f}% (Layer {best_exp.best_layer})",
                f"- **相比基线提升**: {improvement:+.2f}%",
                f"- **训练耗时**: {format_time(best_exp.total_time)}",
                "",
                "### 各层详细结果",
                "",
                "| 层 | 准确率 | 正确/总数 | Gate值 | 相比基线 |",
                "|-----|--------|-----------|--------|----------|",
            ])
            
            for lr in sorted(best_exp.layer_results, key=lambda x: x.accuracy, reverse=True):
                if lr.error:
                    report_lines.append(f"| L{lr.layer} | ERROR | - | - | - |")
                else:
                    diff = lr.accuracy - self.search_results.baseline_accuracy
                    report_lines.append(
                        f"| L{lr.layer} | {lr.accuracy:.2f}% | "
                        f"{lr.correct}/{lr.total} | {lr.gate:.4f} | {diff:+.2f}% |"
                    )
        
        # 所有实验结果
        report_lines.extend([
            "",
            "## 4. 所有实验结果",
            "",
            "按平均准确率排序：",
            "",
            "| 排名 | kl_beta | kl_warmup | lr | 平均准确率 | 最佳层 | 最高准确率 | 状态 |",
            "|------|---------|-----------|-----|-----------|--------|-----------|------|",
        ])
        
        # 按准确率排序
        sorted_experiments = sorted(
            enumerate(self.search_results.experiments),
            key=lambda x: x[1].avg_accuracy if x[1].status == "completed" else -1,
            reverse=True
        )
        
        for rank, (idx, exp) in enumerate(sorted_experiments, 1):
            if exp.status == "completed":
                is_best = " 🏆" if (self.best_params and exp.params == self.best_params) else ""
                report_lines.append(
                    f"| {rank} | {exp.params['kl_beta']} | "
                    f"{exp.params['kl_warmup_steps']} | "
                    f"{exp.params['abc_learning_rate']} | "
                    f"{exp.avg_accuracy:.2f}% | L{exp.best_layer} | "
                    f"{exp.max_accuracy:.2f}% | ✓{is_best} |"
                )
            else:
                report_lines.append(
                    f"| {rank} | {exp.params['kl_beta']} | "
                    f"{exp.params['kl_warmup_steps']} | "
                    f"{exp.params['abc_learning_rate']} | "
                    f"- | - | - | ❌ {exp.error_message[:20] if exp.error_message else 'Failed'}... |"
                )
        
        # 分析与建议
        report_lines.extend([
            "",
            "## 5. 分析与建议",
            "",
        ])
        
        if self.search_results.completed_experiments > 0:
            completed_exps = [e for e in self.search_results.experiments 
                            if e.status == "completed"]
            
            # 分析各参数的影响
            param_analysis = {}
            for param in self.param_grid.keys():
                param_analysis[param] = {}
                for exp in completed_exps:
                    val = exp.params[param]
                    if val not in param_analysis[param]:
                        param_analysis[param][val] = []
                    param_analysis[param][val].append(exp.avg_accuracy)
            
            report_lines.append("### 参数敏感性分析")
            report_lines.append("")
            
            for param, val_accs in param_analysis.items():
                report_lines.append(f"#### {param}")
                report_lines.append("")
                report_lines.append("| 取值 | 平均准确率 | 样本数 |")
                report_lines.append("|------|-----------|--------|")
                
                for val in sorted(val_accs.keys()):
                    accs = val_accs[val]
                    avg = sum(accs) / len(accs)
                    report_lines.append(f"| {val} | {avg:.2f}% | {len(accs)} |")
                
                report_lines.append("")
            
            # 建议
            report_lines.append("### 建议")
            report_lines.append("")
            
            if self.best_avg_accuracy > self.search_results.baseline_accuracy:
                report_lines.append(
                    f"1. **推荐使用最佳配置**: {format_params(self.best_params)}"
                )
                report_lines.append(
                    f"2. **最佳模型已保存到**: `{self.best_output_dir}`"
                )
                report_lines.append(
                    f"3. **预期提升**: 相比基线提升 "
                    f"{self.best_avg_accuracy - self.search_results.baseline_accuracy:+.2f}%"
                )
            else:
                report_lines.append(
                    "⚠️ 当前参数配置未能超越基线，建议：\n"
                    "1. 扩大参数搜索范围\n"
                    "2. 增加训练 epoch\n"
                    "3. 检查数据质量"
                )
        
        report_lines.extend([
            "",
            "---",
            "",
            f"报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        # 写入报告
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("报告已生成")
        self.logger.info("=" * 70)
        self.logger.info(f"JSON 结果: {json_file}")
        self.logger.info(f"中文报告: {report_file}")
        if self.best_avg_accuracy > 0:
            self.logger.info(f"最佳模型: {self.best_output_dir}")
        
        print(f"\n📄 报告已保存到: {report_file}")
        if self.best_avg_accuracy > 0:
            print(f"🏆 最佳模型保存在: {self.best_output_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    best_dir = os.path.join(BEST_OUTPUT_BASE, f"{DATASET}_best")
    
    print("=" * 70)
    print("ABC Vector 超参数自动搜索")
    print("=" * 70)
    print(f"模型: {MODEL_PATH.split('/')[-1]}")
    print(f"数据集: {DATASET}")
    print(f"输出目录: {RESULTS_DIR}")
    print(f"最佳模型目录: {best_dir}")
    print(f"参数组合数: {len(list(product(*PARAM_GRID.values())))}")
    print("=" * 70)
    
    try:
        # 创建搜索器
        searcher = ABCHyperparameterSearch(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
            dataset=DATASET,
            data_path=DATA_PATH,
            results_dir=RESULTS_DIR,
            param_grid=PARAM_GRID,
            fixed_params=FIXED_PARAMS,
            layers=LAYERS,
            best_output_base=BEST_OUTPUT_BASE,
        )
        
        # 运行搜索
        searcher.run_search()
        
        print("\n✅ 超参数搜索完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断搜索")
        send_email(
            subject=f"⚠️ 中断: {DATASET} 超参数搜索",
            body="用户手动中断了超参数搜索。\n中间结果已保存。",
            is_error=True
        )
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ 搜索失败: {e}")
        traceback.print_exc()
        
        send_email(
            subject=f"❌ 失败: {DATASET} 超参数搜索",
            body=f"超参数搜索因错误终止。\n\n错误信息:\n{str(e)}\n\n"
                 f"堆栈跟踪:\n{traceback.format_exc()}",
            is_error=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()