# Crypto V7 Optimized Models

優化後的加密貨幣價格預測模型，基於最新的深度學習研究。

## 改進內容

### 1. 數據優化
- **K 線數量**: 8000 根（約 2-3 個月的數據）
- **時間框架**: 15 分鐘 & 1 小時
- **數據來源**: 優先使用 Binance US API，自動降級到 yfinance

### 2. 模型架構改進 [web:114][web:116][web:119]

#### 基層架構
- **4 層 Bidirectional LSTM** 而非 3 層
  - 第 1 層: 256 units (Bi-LSTM)
  - 第 2 層: 128 units (Bi-LSTM)
  - 第 3 層: 64 units (Bi-LSTM)
  - 第 4 層: 32 units (Bi-LSTM)

- **Layer Normalization**: 每層 LSTM 後添加
  - 穩定訓練過程
  - 減少梯度消失問題

- **正則化**:
  - L1/L2 正則化: 1e-5
  - Dropout: 0.2-0.3
  - 防止過度擬合

#### 技術指標優化 [web:115][web:125]
- **新增指標**:
  - ROC (Rate of Change): 動量指標
  - Volume Normalization: 成交量標準化
  - 總共 14 個特徵

- **序列長度**: 120（而非 60）
  - 捕捉更長期的市場趨勢
  - 改善預測準確度

### 3. 訓練優化 [web:112][web:134]

#### 優化器配置
- **Adam 優化器**:
  - 學習率: 0.0005（更保守）
  - Gradient Clipping: 1.0（防止梯度爆炸）

#### 損失函數
- **加權多輸出損失**:
  - Open 價格: 0.9 倍權重
  - **Close 價格: 1.2 倍權重**（最重要）
  - High 價格: 0.9 倍權重
  - Low 價格: 0.9 倍權重

#### 訓練配置
- **Epochs**: 200（自動停止，最多 20 個 epoch 無改進）
- **Batch Size**: 16
- **驗證分割**: 80/20
- **Early Stopping**: 監控 val_loss
- **Learning Rate 衰減**: factor=0.5, patience=8

### 4. 評估指標

模型評估使用以下指標：
- **MSE** (Mean Squared Error): 預測誤差的平方
- **MAE** (Mean Absolute Error): 平均絕對誤差
- **MAPE** (Mean Absolute Percentage Error): 百分比誤差

## 在 Colab 中使用

### 安裝依賴
```bash
!pip install python-binance tensorflow pandas scikit-learn matplotlib yfinance
```

### 執行優化訓練
```python
cd /content/repo && git pull
python train_v7_optimized.py
```

或使用完整工作流：
```python
# 修改 colab_complete_workflow.py 中的訓練腳本路徑為 train_v7_optimized.py
!python colab_complete_workflow.py
```

### 查看結果
```python
import json

# 查看訓練結果
with open('/content/all_models/metadata_v7_opt.json') as f:
    metadata = json.load(f)
    for key, value in metadata.items():
        print(f"{key}:")
        print(f"  MAPE: {value['val_mape']:.2f}%")
        print(f"  MAE:  {value['val_mae']:.6f}")
        print(f"  Samples: {value['train_samples']} train, {value['val_samples']} val")
```

### 可視化預測
```bash
!python visualize_predictions.py --list-klines

!python visualize_predictions.py \
    --model_path /content/all_models/BTCUSDT_15m_v7_opt.keras \
    --symbol BTCUSDT_15m \
    --output /tmp/btc_15m_opt.png
```

## 研究依據

本優化基於以下最新研究：

1. **Bi-LSTM 優勢** [web:116]
   - Bi-LSTM 在加密貨幣價格預測中表現優於單向 LSTM
   - 捕捉雙向時間依賴性

2. **多層架構** [web:114]
   - LSTM 和 GRU 是最廣泛使用的方法
   - 多層架構改善複雜模式捕捉

3. **正則化策略** [web:112][web:134]
   - Layer Normalization 穩定訓練
   - 適當的 dropout 和 L1/L2 正則化防止過度擬合

4. **技術指標選擇** [web:125]
   - 多種技術指標結合改善預測
   - ROC 和 Bollinger Bands 最有效

5. **序列長度優化** [web:128]
   - 120 步序列長度在加密貨幣預測中表現最佳
   - 平衡計算複雜度和預測準確度

## 預期性能

根據研究論文，此模型應達到：
- **MAPE**: 3-6% （優秀）
- **R² Score**: 0.97-0.99 （非常好）
- **訓練時間**: 20-40 分鐘（GPU）

## 模型文件

訓練完成後生成：
- `*.keras`: 訓練好的模型檔
- `metadata_v7_opt.json`: 訓練元數據
- `*_*_binance_us.csv`: K 線數據

## 下一步

1. 查看可視化結果
2. 上傳模型到 Hugging Face Hub
3. 部署到生產環境進行即時預測
4. 持續監控模型性能並定期重新訓練

## 參考文獻

[web:112] Bayesian Optimization for Bitcoin Price Prediction (2024)
[web:114] Deep Learning in Financial Markets Review (2024)
[web:115] Cryptocurrency Price Prediction with Deep Neural Networks (2024)
[web:116] Predictive Analysis using Bi-LSTM (2024)
[web:119] Hybrid Model LSTM+Bi-LSTM+XGBoost (2024)
[web:125] CryptoPulse: Dual-Prediction Model (2025)
[web:128] Window-sliding LSTM for Cryptocurrency (2021)
[web:134] Optimizing Bitcoin Price Prediction with LSTM (2024)
