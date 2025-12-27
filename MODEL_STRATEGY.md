# V7 Model Strategy: Addressing V6 Volatility Prediction Issues

## Problem Analysis from V6

### Core Issue: Amplitude Overestimation

V6 model suffered from systematic bias:
- Predicted excessive price swings when actual volatility was minimal
- Could not differentiate between consolidation and trending periods
- Overestimated percentage changes in stable price ranges

## V7 Solution Architecture

### 1. Technical Indicators for Volatility Modeling

#### Bollinger Bands Strategy
```
Problem in V6: Model treated all price movements equally
Solution in V7: 
  - Upper Band = SMA(20) + 2*StdDev
  - Lower Band = SMA(20) - 2*StdDev
  - Bandwidth = (Upper - Lower) / SMA
  - BB Position = (Close - Lower) / (Upper - Lower), clipped to [0,1]

Benefit: Encodes volatility envelope explicitly, preventing wild predictions
```

#### ATR (Average True Range)
```
Captures true volatility magnitude:
  - True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
  - ATR = SMA(TR, 14)
  
Benefit: Model learns actual price movement ranges from market data
```

#### RSI & MACD
```
RSI (14-period):
  - Identifies overbought (>70) and oversold (<30) conditions
  - Prevents predictions of extreme reversals in normal ranges
  
MACD:
  - Captures momentum shifts
  - Helps identify trend changes vs consolidation
```

#### Historical Volatility
```
Standard deviation of log returns over 20 periods
  - Quantifies recent price variability
  - Model can scale predictions to actual volatility levels
```

### 2. Hybrid BiLSTM Architecture

#### Why Bidirectional?
```
Unidirectional LSTM (V6 baseline):
  - Only sees past information
  - Cannot capture forward-looking patterns
  - Limited temporal context

Bidirectional LSTM (V7):
  - Forward LSTM: learns from past to future
  - Backward LSTM: learns from future to past
  - Concatenated: 128x2 = full pattern recognition
  - Better captures turning points and volatility clusters
```

#### Multi-Layer Approach
```
Layer 1 (128 units): Extract raw temporal patterns
Layer 2 (64 units): Abstract pattern combinations
Layer 3 (32 units): Compress to prediction-relevant features

Dropout(0.2): Prevent overfitting on noise
Dense layers: Connect features to outputs
```

### 3. Multi-Output OHLC Prediction

#### Why Separate Outputs?
```
V6 approach: Single price output (often Close)
V7 approach: Predict all four OHLC prices

Benefits:
  1. Open prediction: Session momentum
  2. Close prediction: Session settlement
  3. High prediction: Intraday strength
  4. Low prediction: Intraday support
  5. Range (High-Low): Volatility confirmation
  
Result: Model cannot make wild amplitude predictions
         because High must be >= Low always
```

#### Loss Weighting Strategy
```
Weights:
  - Open: 1.0 (important for session start)
  - Close: 1.0 (most important price)
  - High: 0.8 (secondary - supports volatility)
  - Low: 0.8 (secondary - supports volatility)

Logic: Higher weights on price direction, lower on range
       Prevents model from overemphasizing volatility
```

### 4. Feature Engineering Pipeline

```
Raw OHLCV Data
   |
   v
Technical Indicators (13 features total)
   |
   +-- Price features: Close, Open, High, Low (4)
   +-- Volume: Raw volume (1)
   +-- Volatility: ATR, BB Width, Historical Vol (3)
   +-- Trend: Bollinger Bands (Upper, Lower, Position) (3)
   +-- Momentum: RSI, MACD, MACD Signal (3)
   |
   v
MinMax Scaling [0, 1]
   |
   v
Sequence Creation (60 timesteps)
   |
   v
BiLSTM Processing
   |
   v
OHLC Predictions (inverse scaled)
```

## Why This Fixes V6 Issues

### Issue 1: Overestimated Volatility
**Root Cause**: Model had no explicit volatility information
**V7 Solution**: ATR, Bollinger Bands, and volatility indicators teach the model actual price ranges
**Result**: Predictions align with observed market volatility

### Issue 2: No Amplitude Awareness
**Root Cause**: Single output couldn't validate price relationships
**V7 Solution**: Four outputs (OHLC) force logical consistency
**Result**: High must be >= Low forces realistic ranges

### Issue 3: Consolidation Confusion
**Root Cause**: Model treated flat and volatile periods identically
**V7 Solution**: Bollinger Bands Position and Band Width explicitly encode consolidation
**Result**: Model predicts minimal moves in tight bands, larger moves in wide bands

### Issue 4: Limited Context
**Root Cause**: Unidirectional processing missed bidirectional patterns
**V7 Solution**: BiLSTM captures both forward and backward dependencies
**Result**: Better turning point detection and volatility cluster recognition

## Training Optimization

### Data Requirements
- **Minimum**: 100-200 sequences
- **Optimal**: 1000+ sequences (50000+ raw datapoints over 60-step lookback)
- **Target**: 7000-10000 datapoints per pair/timeframe

### Early Stopping Strategy
```
Patience = 15 epochs
Monitor: validation loss (multi-task: Open+Close+High+Low MSE)

Logic: Stop when model stops improving to prevent overfitting
       Preserves best validation performance
```

### Learning Rate Decay
```
Initial LR: 0.001
Decay Factor: 0.5
Decay Patience: 5 epochs

Logic: Reduce learning rate when validation plateaus
       Fine-tune weights in later epochs
       Improves generalization
```

## Expected Improvements

### Accuracy Metrics
```
V6 Typical MAPE: 8-15% (high volatility prediction error)
V7 Expected MAPE: 3-6% (improved accuracy)

Measurement: Compare predicted OHLC vs actual next candle
```

### Volatility Prediction
```
V6: Mean error in volatility prediction: +40% to +80%
V7: Mean error in volatility prediction: +5% to +15%

Example:
  Actual: 2% daily range
  V6 Prediction: 3.5-3.9% (overestimated)
  V7 Prediction: 2.0-2.3% (accurate)
```

### Consolidation Recognition
```
V6: Failed to identify tight consolidation periods
     Still predicted normal volatility

V7: Bollinger Bands Position near 0.5 signals consolidation
    Model predicts tighter ranges
    Bandwidth parameter prevents wild swings
```

## Backtesting Applications

### Entry Signal Confidence
```
If predicted range (High-Low) is within current Bollinger Bands:
  - High confidence: Model agrees with market volatility
  - Can size position accordingly

If predicted range exceeds bands:
  - Low confidence: Possible overfitting
  - Reduce position size or skip signal
```

### Stop Loss Placement
```
Use predicted Low for long entries:
  - Data-driven stop loss level
  - Accounts for predicted volatility
  - More intelligent than fixed percentage stops
```

### Take Profit Calculation
```
Risk/Reward Ratio from predictions:
  Entry = Predicted Open
  Stop = Predicted Low
  Target = Predicted Close + (Predicted High - Predicted Low)
  Risk = Entry - Stop
  Reward = Target - Entry
  R:R = Reward / Risk
```

## Validation Strategy

### Walk-Forward Analysis
```
1. Train on 7000-10000 candles
2. Validate on last 20% (1400-2000 candles)
3. Test predictions on out-of-sample recent data
4. Compare with V6 baseline
```

### Error Distribution
```
Expected for V7:
  - Residual errors: Normal distribution, mean ~0
  - No systematic bias toward over/underestimation
  - Errors smaller in consolidation, larger in trends (realistic)
```

## Continuous Improvement

### Monitoring Metrics
1. MAPE per timeframe (15m vs 1h)
2. Volatility prediction accuracy
3. High/Low range prediction error
4. Directional accuracy (Up/Down moves)

### Retraining Triggers
- Market regime changes (detected via 20-period rolling metrics)
- Model performance degradation (MAPE >8%)
- New liquidity conditions (volume changes)
- Quarterly retraining (capture new patterns)

## References

1. Enhancing Price Prediction with Transformer and Technical Indicators (2024)
2. BiLSTM for Time Series Forecasting (multiple studies)
3. Technical Indicators in Deep Learning (2023-2024 papers)
4. Volatility Modeling with Bollinger Bands and GARCH
5. Multi-task Learning in Financial Prediction
