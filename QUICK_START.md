# Quick Start: V7 Model Training in Colab

## Copy-Paste Instructions for Colab

Open https://colab.research.google.com and run these cells in order:

### Cell 1: Setup (1 minute)
```python
import sys
print('Python:', sys.version)
!nvidia-smi | grep -E 'GPU|Memory'
print('Setup Ready')
```

### Cell 2: Install Dependencies (3-5 minutes)
```python
import subprocess

packages = [
    'tensorflow>=2.14',
    'numpy',
    'pandas',
    'scikit-learn',
    'yfinance',
    'python-binance',
    'huggingface-hub'
]

for pkg in packages:
    print(f'Installing {pkg}...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], 
                   capture_output=True)

import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print('GPU:', tf.config.list_physical_devices('GPU'))
```

### Cell 3: Clone and Prepare (1 minute)
```python
!git clone https://github.com/caizongxun/crypto-v7-models.git /content/repo 2>/dev/null || echo 'Already cloned'
!mkdir -p /content/all_models
print('Repository ready at /content/repo')
print('Output directory: /content/all_models')
```

### Cell 4: Load Training Code (10 seconds)
```python
exec(open('/content/repo/train_v7_main.py').read())
print('Training classes loaded successfully')
```

### Cell 5: Start Training (1.5-2.5 hours)
```python
print('Starting training for 20 cryptocurrencies x 2 timeframes = 40 models')
print('Estimated time: 1.5-2.5 hours\n')

pipeline = TrainingPipeline(output_dir='/content/all_models')
metadata = pipeline.train_all_models()

print('\nTraining Complete!')
import os
models = [f for f in os.listdir('/content/all_models') if f.endswith('.h5')]
print(f'Models trained: {len(models)}')
```

### Cell 6: Verify Models (10 seconds)
```python
import os
import json

model_files = sorted([f for f in os.listdir('/content/all_models') if f.endswith('.h5')])
print(f'Total models: {len(model_files)}\n')

total_size = 0
for model in model_files:
    size_mb = os.path.getsize(f'/content/all_models/{model}') / 1024 / 1024
    total_size += size_mb
    if model.endswith('_v7.h5'):
        symbol, timeframe = model.replace('_v7.h5', '').rsplit('_', 1)
        print(f'{symbol:12} {timeframe:3} - {size_mb:.2f} MB')

metadata_size = os.path.getsize('/content/all_models/metadata_v7.json') / 1024
print(f'\nMetadata: {metadata_size:.1f} KB')
print(f'Total size: {total_size:.1f} MB')
```

### Cell 7: Upload to Hugging Face (5 minutes)
```python
from huggingface_hub import upload_folder
import getpass

print('Preparing upload to Hugging Face...')
print('Dataset: zongowo111/cpb-models')
print('Target folder: models_v7\n')

token = getpass.getpass('Enter Hugging Face token: ')

try:
    print('Starting upload...')
    upload_folder(
        folder_path='/content/all_models',
        repo_id='zongowo111/cpb-models',
        path_in_repo='models_v7',
        repo_type='dataset',
        token=token,
        multi_commits=True
    )
    print('\nUpload successful!')
    print('Models available at:')
    print('https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v7')
except Exception as e:
    print(f'Upload error: {e}')
    print('Models saved locally at /content/all_models/')
```

---

## Understanding the Output

### During Training
```
====== Training BTCUSDT 15m ======
Data shape: (8500, 6)
Training samples: 6800, Validation samples: 1700
Epoch 1/100
32/214 [===>..................] - loss: 0.0456 - close_loss: 0.0412
Epoch 45/100  (early stopped)
Model saved: /content/all_models/BTCUSDT_15m_v7.h5
Validation Metrics - MSE: 0.000125, MAE: 0.008234, MAPE: 2.34%
```

**Interpretation**:
- MAPE 2.34% = Prediction accuracy within 2.34% of actual
- Early stopping at epoch 45 = Model converged quickly
- Successful save = Model ready for prediction

### After Training
```
Total models: 40
ADUSDT         15m - 4.23 MB
ADUSDT         1h  - 4.23 MB
APEUSDT        15m - 4.23 MB
...
Metadata: 356.2 KB
Total size: 175.3 MB
```

**Meaning**:
- 40 models = All cryptocurrencies + timeframes trained
- Consistent 4.23 MB = Same architecture for all
- Total 175 MB = Single upload to Hugging Face

---

## Troubleshooting Quick Fixes

### GPU Not Available
```python
# Enable GPU
# Menu: Runtime -> Change runtime type -> GPU (T4)
# Then restart cells
```

### Out of Memory
```python
# In Cell 4, modify before executing:
# Find: model.train(..., epochs=100, batch_size=32)
# Change to: batch_size=16 or batch_size=8
```

### Network Timeout During Upload
```python
# Wait 5 minutes, then retry Cell 7
# Or download locally and upload manually later
```

### Insufficient Data for Some Cryptos
```
Output: "Insufficient data for XRPUSDT 15m"
This is normal - that pair will be skipped
Remaining pairs continue training normally
```

---

## What Gets Trained

### 20 Cryptocurrencies
```
BTC, ETH, BNB, XRP, ADA, DOGE, SOL, LINK, MATIC, AVAX,
UNI, LTC, BCH, ETC, XLM, VET, FIL, THETA, NEAR, APE
```

### 2 Timeframes Each
```
15-minute candles (suitable for scalping/day trading)
1-hour candles (suitable for swing trading)
```

### Total Models: 40
```
20 cryptos x 2 timeframes = 40 models
```

---

## Expected Performance

### Training Speed
- Per model: 2-5 minutes
- Total 40 models: 1.5-2.5 hours

### Model Accuracy (typical)
- MAPE: 2-5% (excellent)
- MSE: < 0.0001 (minimal)
- MAE: < 0.01 (small errors)

### File Sizes
- Per model: ~4.2 MB
- All 40 + metadata: ~175 MB
- Fits in Hugging Face limits

---

## After Training

### Models Location
```
Hugging Face: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v7
```

### Using Models
```python
import tensorflow as tf

model = tf.keras.models.load_model('BTCUSDT_15m_v7.h5')
predictions = model.predict(X_test)

open_price = predictions[0]
close_price = predictions[1]
high_price = predictions[2]
low_price = predictions[3]
```

### Training Data Size
- Training samples per model: ~6000-7000
- Validation samples per model: ~1500-1700
- Sequence length: 60 timesteps
- Features: 13 technical indicators

---

## Key Differences from V6

### V6
- Single output (close price)
- Basic features (OHLCV only)
- Often overestimated volatility

### V7 Improvements
- 4 outputs (Open, High, Low, Close)
- 13 technical indicators
- Better volatility accuracy
- BiLSTM bidirectional processing
- Early stopping to prevent overfitting

---

## Next Steps

1. **Run the training** in Colab (follow cells above)
2. **Upload to Hugging Face** with token
3. **Download models** locally for testing
4. **Backtest** predictions on recent data
5. **Deploy** in trading strategies

---

## Still Have Questions?

- README.md: Architecture and features
- MODEL_STRATEGY.md: Why each component improves accuracy
- COLAB_GUIDE.md: Detailed troubleshooting

GitHub Repo: https://github.com/caizongxun/crypto-v7-models
