# Cryptocurrency V7 Model Training Repository

Advanced cryptocurrency price prediction model using Hybrid BiLSTM architecture with technical indicators for improved volatility and amplitude accuracy.

## Version 7 Improvements

Compared to v6, this version addresses critical accuracy issues:

1. **Volatility Prediction Enhancement**
   - Implements Bollinger Bands decomposition for accurate volatility modeling
   - Uses ATR (Average True Range) indicator for volatility magnitude
   - Calculates bandwidth percentage to capture price action ranges

2. **Amplitude Accuracy**
   - Prevents overestimation of price changes
   - Models actual volatility patterns through technical indicators
   - Separate high/low predictions to capture full candle range

3. **Hybrid BiLSTM Architecture**
   - Bidirectional processing captures both past and future patterns
   - Multi-layer LSTM with dropout regularization
   - Attention mechanism through concatenated forward-backward sequences

4. **Technical Indicators Integration**
   - Bollinger Bands (upper, middle, lower, bandwidth)
   - ATR (Average True Range)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Historical Volatility
   - Bollinger Bands Position indicator

## Features

- **Multi-Timeframe Support**: 15-minute and 1-hour candles
- **20+ Cryptocurrencies**: BTC, ETH, BNB, XRP, ADA, DOGE, SOL, LINK, MATIC, AVAX, UNI, LTC, BCH, ETC, XLM, VET, FIL, THETA, NEAR, APE
- **Multi-Source Data**: Binance API and Yahoo Finance
- **OHLC Prediction**: Simultaneous prediction of Open, High, Low, Close prices
- **GPU Optimized**: Full TensorFlow 2.14+ compatibility
- **Hugging Face Integration**: Direct model upload to dataset repository

## Training Architecture

### Model Structure

```
Input (60 steps x 13 features)
  |
  v
BiLSTM Layer 1 (128 units, return_sequences=True)
  |
  Dropout(0.2)
  |
  v
BiLSTM Layer 2 (64 units, return_sequences=True)
  |
  Dropout(0.2)
  |
  v
BiLSTM Layer 3 (32 units, return_sequences=False)
  |
  Dropout(0.2)
  |
  v
Dense Layer (64 units, ReLU)
  |
  Dense Layer (32 units, ReLU)
  |
  +----> Open Output (Dense 1)
  +----> Close Output (Dense 1)
  +----> High Output (Dense 1)
  +----> Low Output (Dense 1)
```

### Technical Indicators (13 Features)

1. Close Price
2. Open Price
3. High Price
4. Low Price
5. Volume
6. ATR (14-period)
7. Bollinger Bands Upper
8. Bollinger Bands Lower
9. Bollinger Bands Width
10. RSI (14-period)
11. MACD
12. Volatility (20-period historical)
13. Bollinger Bands Position (normalized 0-1)

## Dataset Specifications

- **Data Points**: 7,000-10,000 per cryptocurrency per timeframe
- **Preprocessing**: MinMax normalization (0-1 range)
- **Sequence Length**: 60 timesteps
- **Train/Val Split**: 80/20
- **Data Source**: Binance US API, Yahoo Finance fallback

## Training Hyperparameters

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: MSE with weighted outputs
  - Open, Close: 1.0 weight
  - High, Low: 0.8 weight
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Early Stopping**: patience=15 on validation loss
- **Learning Rate Decay**: factor=0.5, patience=5

## Setup Instructions

### Local Setup

```bash
git clone https://github.com/caizongxun/crypto-v7-models.git
cd crypto-v7-models
pip install -r requirements.txt
```

### Google Colab Setup

1. Open Google Colab
2. Execute the provided `colab_train.ipynb`
3. Follow the cell-by-cell workflow

## Colab Training Workflow

### Step 1: Environment Setup
- Install TensorFlow 2.14+, dependencies
- Verify GPU availability

### Step 2: Data Acquisition
- Clone repository
- Create output directories

### Step 3: Model Training
- Fetch data from Binance US and Yahoo Finance
- Calculate technical indicators
- Preprocess and normalize
- Train BiLSTM models (20+ cryptocurrencies x 2 timeframes)

### Step 4: Model Storage
- All models saved to `/content/all_models/`
- Format: `{SYMBOL}_{TIMEFRAME}_v7.h5`
- Metadata saved as `metadata_v7.json`

### Step 5: Hugging Face Upload
- Upload entire `all_models` folder
- Destination: `models_v7` subfolder in dataset
- Token required (provided via getpass prompt)

## File Structure

```
crypto-v7-models/
training_pipeline/
train_v7_main.py           # Main training script
colab_train.ipynb          # Colab notebook
requirements.txt           # Python dependencies
README.md                  # This file
```

## Output Structure

After training:

```
all_models/
BTCUSDT_15m_v7.h5
BTCUSDT_1h_v7.h5
ETHUSDT_15m_v7.h5
ETHUSDT_1h_v7.h5
...
metadata_v7.json
```

## Hugging Face Repository

- **Dataset Repo**: https://huggingface.co/datasets/zongowo111/cpb-models
- **Models Folder**: `models_v7`

## Model Performance Metrics

Each model evaluation includes:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

Metrics stored in `metadata_v7.json` for each symbol/timeframe combination.

## Model Inference

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('BTCUSDT_15m_v7.h5')
predictions = model.predict(X_test)

open_pred = predictions[0]
close_pred = predictions[1]
high_pred = predictions[2]
low_pred = predictions[3]
```

## Key Advantages Over V6

1. **Improved Amplitude Modeling**: Technical indicators capture actual volatility patterns
2. **Better Volatility Prediction**: Bollinger Bands decomposition prevents overestimation
3. **Bidirectional Processing**: BiLSTM captures both backward and forward dependencies
4. **Multiple Price Points**: Separate OHLC predictions for comprehensive candle modeling
5. **Robust Feature Set**: 13 carefully selected indicators prevent overfitting

## Troubleshooting

### Insufficient Data
- If crypto has <100 samples: model skips this pair
- Solution: Check data source availability

### GPU Out of Memory
- Reduce batch size from 32 to 16 or 8
- Reduce sequence length from 60 to 30

### Binance Connection Issues
- Falls back to Yahoo Finance automatically
- Check network connectivity

### TA-Lib Installation
- Use: `pip install TA-Lib` for binary wheel
- Alternative: `pip install ta-lib` (source build)

## Future Enhancements

- Ensemble methods combining multiple model outputs
- Transformer architecture for attention mechanisms
- Real-time prediction API deployment
- Portfolio optimization using predicted prices
- Sentiment analysis integration

## Citation

Based on research in:
- BiLSTM for time series forecasting
- Hybrid approaches combining technical indicators with deep learning
- Cryptocurrency price prediction with volatility modeling

## License

MIT License

## Contact

Repository: https://github.com/caizongxun/crypto-v7-models
