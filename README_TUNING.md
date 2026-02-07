# ABC Vector 超参数自动搜索使用指南

## 1. 文件说明

| 文件 | 说明 |
|------|------|
| `abc_hyperparameter_search.py` | 主调参脚本 |
| `email_helper.py` | 邮件配置模块 |

## 2. 快速开始

### 2.1 修改配置

在 `abc_hyperparameter_search.py` 文件开头修改以下配置：

```python
# 模型配置
MODEL_PATH = "/path/to/your/model"  # 修改为你的模型路径
MODEL_NAME = "qwen"  # "qwen" 或 "llama"

# 数据集配置
DATASET = "gsm8k"  # "gsm8k", "math_easy", "math_hard", "mmlu_pro"
DATA_PATH = "/path/to/your/data"  # 修改为你的数据路径

# 输出路径
RESULTS_DIR = "./results"
```

### 2.2 配置邮件通知（可选）

如果需要邮件通知，请修改 `email_helper.py`：

**使用 Gmail（推荐）：**
1. 在 Gmail 设置中开启两步验证
2. 生成应用专用密码：Google账户 → 安全性 → 两步验证 → 应用专用密码
3. 修改 `email_helper.py`：

```python
GMAIL_CONFIG = {
    "enabled": True,
    "email": "your.email@gmail.com",
    "app_password": "xxxx xxxx xxxx xxxx",
}
```

4. 测试邮件：
```bash
python email_helper.py --test
```

### 2.3 运行调参

```bash
# 使用 nohup 后台运行（推荐）
nohup python abc_hyperparameter_search.py > tuning.log 2>&1 &

# 或直接运行
python abc_hyperparameter_search.py
```

## 3. 参数配置说明

### 3.1 搜索空间（可修改）

```python
PARAM_GRID = {
    "kl_beta": [0.1, 0.5, 1.0, 2.0, 5.0],
    "kl_warmup_steps": [0, 50, 100, 200, 500],
    "abc_learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}
```

总实验数 = 5 × 5 × 5 = **125 个参数组合**

### 3.2 固定参数

```python
FIXED_PARAMS = {
    "num_epochs": 10,          # 训练轮数
    "batch_size": 2,           # 批大小
    "gradient_accumulation_steps": 2,
    "abc_hidden_dim": 512,     # ABC网络隐藏层维度
    "sigma_min": 1e-4,
    "max_length": 1024,
    "warmup_ratio": 0.1,
    "weight_decay": 1e-3,
    "num_support_samples": 3000,  # 支持集大小
    "num_test_samples": 100,      # 测试集大小
    "max_new_tokens": 512,
    "num_beams": 3,
}
```

### 3.3 测试层配置

```python
LAYERS = list(range(0, 27, 2))  # 0, 2, 4, ..., 26
```

## 4. 输出文件

运行后，结果保存在 `results/{dataset}/` 目录下：

| 文件 | 说明 |
|------|------|
| `abc_tuning_{dataset}_{timestamp}.log` | 详细日志 |
| `abc_tuning_results_{timestamp}.json` | JSON格式结果 |
| `abc_tuning_report_{timestamp}.md` | 中文分析报告 |

## 5. 报告内容

生成的报告包含：
1. **实验配置** - 参数搜索空间和固定参数
2. **最佳配置** - 最优参数组合及详细结果
3. **所有实验结果** - 按准确率排序的完整列表
4. **分析与建议** - 参数敏感性分析和改进建议

## 6. 错误处理

- **CUDA OOM**: 脚本会自动清理显存并跳过当前实验
- **其他错误**: 记录到日志并发送邮件通知
- **中断恢复**: 中间结果自动保存到 `*_intermediate_*.json`

## 7. 时间估算

对于单个参数配置（14层 × 10 epochs）：
- H100: 约 15-30 分钟
- A40: 约 30-60 分钟

完整搜索（125 个配置）：
- H100: 约 31-62 小时
- A40: 约 62-125 小时

**建议**: 先用较小的参数空间测试，确认无误后再运行完整搜索。

## 8. 常见问题

### Q: 如何缩小搜索空间？

修改 `PARAM_GRID`，减少每个参数的取值数量：

```python
PARAM_GRID = {
    "kl_beta": [0.5, 1.0, 2.0],      # 3个值
    "kl_warmup_steps": [0, 100],      # 2个值
    "abc_learning_rate": [1e-4, 5e-4], # 2个值
}
# 总实验数: 3 × 2 × 2 = 12
```

### Q: 如何只测试特定层？

修改 `LAYERS`：

```python
LAYERS = [10, 12, 14]  # 只测试这三层
```

### Q: 如何禁用邮件通知？

在 `email_helper.py` 中设置：

```python
EMAIL_ENABLED = False
```

### Q: 如何查看运行进度？

```bash
# 查看后台运行的日志
tail -f tuning.log

# 或查看 results 目录下的日志文件
tail -f results/gsm8k/abc_tuning_*.log
```

## 9. 两台服务器配置建议

**H100 服务器**（更快）：
- 使用较大的 batch_size
- 可运行更密集的参数网格

**A40 服务器**（显存较小）：
- 保持 batch_size=2
- 可能需要减少 max_length

如需并行运行，可以将参数空间分割：

```python
# 服务器1
PARAM_GRID = {
    "kl_beta": [0.1, 0.5, 1.0],  # 前半部分
    ...
}

# 服务器2
PARAM_GRID = {
    "kl_beta": [2.0, 5.0],  # 后半部分
    ...
}
```