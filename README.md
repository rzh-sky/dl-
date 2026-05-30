# 大作业代码基线（详细说明）

本项目提供一个**完整可运行**的示例流程：  
**数据预处理 → 特征构造 → 时序模型训练 → 验证集回测**。

---

## 0. 目录结构

- `A股数据/`：原始数据（daily / metric / moneyflow / stock_st / basic.csv）
- `src/`：核心代码
  - `data_preprocess.py`：数据读取、特征构造、标签生成
  - `dataset.py`：滑窗样本生成
  - `model.py`：LSTM/GRU/TCN/Transformer 回归模型
  - `train.py`：训练 + 验证评估（Loss）
  - `backtest.py`：简单回测策略
  - `run_all.py`：一键流程
  - `config.py`：统一参数配置
- `outputs/`：输出目录（自动生成）
  - `processed/`：特征文件
  - `preds/`：预测结果
  - `backtest/`：回测结果

---

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 2. 数据来源与使用范围

**使用数据：**
- 日线量价（`A股数据/daily/*.csv`）
- ST 标记（`A股数据/stock_st/*.csv`）
- 股票列表（`A股数据/basic.csv`）
- 基本面指标（`A股数据/metric/*.csv`）
- 资金流向（`A股数据/moneyflow/*.csv`）

**股票池**：
- 默认：**非北交所 + 剔除每日 ST** 的全部 A 股  
- 可用 `--max-stocks` 抽样子集快速测试

**时间范围**：
- 默认：`20190101` 到 `20251231`（可配置）

---

## 3. 特征工程（低内存）

### 3.1 基础特征（BASE_FEATURE_COLS）
当前实际计算的基础特征：

- 收益类：`ret1/ret3/ret5/ret10/ret20`
- 量价变化：`vol_chg/amount_chg/vol_chg5/amount_chg5`
- 价差与动量：`hl_range/oc_ret/mom5`
- 均线与波动：`ma5/ma10/ma20/std5/std10`
- RSI：`rsi6/rsi12`

> 代码会按 `src/config.py` 的 `BASE_FEATURE_COLS` 进行派生。如果某些基础列未在原始数据中出现（例如部分基本面/资金流字段），会自动补 0。

### 3.2 派生特征（按交易日截面）
对每个基础特征生成 3 个变体：
- `_z`：z-score 标准化
- `_rank`：截面排名（0~1）
- `_miss`：缺失标记（0/1）

最终特征列为：  
`原始特征 + 原始特征_z + 原始特征_rank + 原始特征_miss`

---

## 4. 预处理与标签

### 4.1 标签定义
- 预测目标：未来 `HORIZON` 日收益率（默认 1 日）

### 4.2 滑动窗口
- 输入：过去 `LOOKBACK` 天的特征序列（默认 20 天）
- 输出：未来 `HORIZON` 日收益

---

## 5. 训练设置

模型对比（可选）：
- LSTM（基线）
- GRU（轻量稳健）
- TCN（卷积时序）
- Transformer（长依赖）
- 训练区间：`TRAIN_START ~ TRAIN_END`
- 验证区间：`VAL_START ~ VAL_END`
- 评估指标：
  - `Loss`（MSE）

> 当前实现仅输出 `Loss`，未计算 `IC/ICIR`。

---

## 6. 回测策略（简单基线）

- 每日按预测排序
- 持仓 `N_HOLD`（默认 10）
- 调仓 `K_TRADE`（默认 2）
- 输出：`outputs/backtest/backtest.csv`

---

## 7. 运行方式

### 7.1 预处理
```bash
python -m src.data_preprocess --start 20190101 --end 20251231 --out outputs/processed/features.parquet
```

快速测试：
```bash
python -m src.data_preprocess --start 20190101 --end 20251231 --max-stocks 200
```

### 7.2 训练
```bash
python -m src.train --data outputs/processed/features.parquet --model lstm
```

对比不同模型：
```bash
python -m src.train --data outputs/processed/features.parquet --model gru
python -m src.train --data outputs/processed/features.parquet --model tcn
python -m src.train --data outputs/processed/features.parquet --model transformer
```

输出：
- 验证集预测：`outputs/preds/val_predictions.csv`

### 7.3 回测
```bash
python -m src.backtest --pred outputs/preds/val_predictions.csv
```

### 7.4 一键流程
```bash
python -m src.run_all --start 20190101 --end 20251231
```

对比所有模型：
```bash
python -m src.run_all --compare
```

---

## 8. 配置修改

主要参数在 `src/config.py`：
- 时间范围：`DEFAULT_START / DEFAULT_END`
- 训练/验证分割：`TRAIN_START / TRAIN_END / VAL_START / VAL_END`
- 窗口与标签：`LOOKBACK / HORIZON`
- 模型超参：`BATCH_SIZE / EPOCHS / LR / HIDDEN / NUM_LAYERS / DROPOUT`
- 回测参数：`N_HOLD / K_TRADE`

---

## 9. 结果输出一览

- 特征文件：`outputs/processed/features.parquet`
- 预测结果：`outputs/preds/val_predictions.csv`（或 `val_predictions_{model}.csv`）
- 回测结果：`outputs/backtest/backtest.csv`（或 `backtest_{model}.csv`）

---

如需加入更多特征、模型或回测逻辑，可在 `src/data_preprocess.py` / `src/model.py` / `src/backtest.py` 中扩展。
