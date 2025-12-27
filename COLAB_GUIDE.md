# Google Colab Training Guide for Crypto V7 Models

## Prerequisites

1. Google Account with access to Google Colab
2. Hugging Face Account
3. Hugging Face API Token

## Access Colab Notebook

1. Open Google Colab: https://colab.research.google.com/
2. Click File -> Open notebook -> GitHub tab
3. Enter: `caizongxun/crypto-v7-models`
4. Select `colab_train.ipynb`
5. Or directly open: https://colab.research.google.com/github/caizongxun/crypto-v7-models/blob/main/colab_train.ipynb

## Complete Training Workflow

### Step 1: GPU Setup Verification

**Cell 1: Check GPU**
```python
!nvidia-smi
```

Expected output:
```
GPU 0: NVIDIA Tesla T4 (or similar)
Memory: 15GB (typical for Colab)
```

**What this means**: GPU acceleration enabled for fast training.

---

### Step 2: Environment Setup

**Cell 2: Install Dependencies**
```python
!pip install -q tensorflow>=2.14
!pip install -q numpy pandas scikit-learn
!pip install -q yfinance
!pip install -q python-binance
!pip install -q ta-lib 2>/dev/null || pip install -q TA-Lib
!pip install -q huggingface-hub
```

**Installation time**: 3-5 minutes

**Verify TensorFlow**:
```python
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
```

Expected:
```
TensorFlow: 2.14.0 (or higher)
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

### Step 3: Repository and Output Setup

**Cell 3: Clone Repository**
```python
!git clone https://github.com/caizongxun/crypto-v7-models.git /content/crypto-v7-models 2>/dev/null || echo 'Repo already cloned'
%cd /content/crypto-v7-models
!git pull
```

**What this does**:
- Downloads the latest training code
- Changes working directory to repo
- Updates to latest version

**Cell 4: Create Output Directory**
```python
!mkdir -p /content/all_models
print('All models directory created at /content/all_models')
```

**Important**: All trained models go here

---

### Step 4: Load Training Script

**Cell 5: Execute Training Script**
```python
exec(open('/content/crypto-v7-models/train_v7_main.py').read())
```

This loads all training classes:
- `CryptoDataFetcher`: Data acquisition
- `TechnicalIndicators`: Feature engineering
- `DataPreprocessor`: Data normalization
- `CryptoV7Model`: BiLSTM architecture
- `TrainingPipeline`: Main orchestrator

---

### Step 5: Start Training

**Cell 6: Begin Training Pipeline**
```python
pipeline = TrainingPipeline(output_dir='/content/all_models')
metadata = pipeline.train_all_models()
```

**Training sequence**:
1. Fetch data for each cryptocurrency pair
2. Calculate 13 technical indicators
3. Preprocess and normalize data
4. Create sequences (60-timestep lookback)
5. Train BiLSTM model (100 epochs with early stopping)
6. Save model to `/content/all_models/`
7. Repeat for all 20 cryptocurrencies x 2 timeframes

**Expected training time**: 1.5-2.5 hours on T4 GPU

**Progress indicators**:
```
====== Training BTCUSDT 15m ======
Data shape: (8500, 6)
Training samples: 6800, Validation samples: 1700
Epoch 1/100
 Epoch 1 | loss: 0.0245 | val_loss: 0.0198
 Epoch 2 | loss: 0.0189 | val_loss: 0.0156
 ...
 Model saved: /content/all_models/BTCUSDT_15m_v7.h5
Validation Metrics - MSE: 0.000125, MAE: 0.008234, MAPE: 2.34%
```

---

### Step 6: Verify Trained Models

**Cell 7: Check Output Files**
```python
import os
model_files = os.listdir('/content/all_models')
print(f'Total models trained: {len([f for f in model_files if f.endswith(".h5")])}')
print('Files in all_models directory:')
for f in sorted(model_files):
    size = os.path.getsize(f'/content/all_models/{f}')
    print(f'  {f}: {size / 1024 / 1024:.2f} MB')
```

Expected output:
```
Total models trained: 40
Files in all_models directory:
  ADAUSDT_15m_v7.h5: 4.23 MB
  ADAUSDT_1h_v7.h5: 4.23 MB
  APEUSDT_15m_v7.h5: 4.23 MB
  ...
  metadata_v7.json: 0.35 MB
```

**Total size**: ~160-180 MB for all 40 models + metadata

---

### Step 7: Hugging Face Upload

**Cell 8: Input API Token**
```python
from huggingface_hub import HfApi
import getpass

hf_token = getpass.getpass('Enter your Hugging Face token: ')
```

**How to get token**:
1. Visit https://huggingface.co/settings/tokens
2. Click "New token"
3. Set permission to "Write"
4. Copy token
5. Paste in Colab input box (hidden input for security)

**Cell 9: Upload to Hugging Face**
```python
from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = 'zongowo111/cpb-models'

try:
    upload_folder(
        folder_path='/content/all_models',
        repo_id=repo_id,
        path_in_repo='models_v7',
        repo_type='dataset',
        token=hf_token,
        multi_commits=True,
        multi_commits_strategy='save_by_pattern'
    )
    print('Upload completed successfully')
except Exception as e:
    print(f'Upload error: {e}')
```

**Upload process**:
1. Creates `models_v7` folder in dataset repo
2. Uploads all `.h5` files
3. Uploads `metadata_v7.json`
4. Uses multiple commits to avoid API rate limits
5. Shows progress indicators

**Upload time**: 3-8 minutes (160MB)

**Result**:
```
Upload completed successfully
```

Models now available at: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v7

---

## Monitoring Training Progress

### Real-time Metrics

During training, observe:

**Loss values** (should decrease):
```
Epoch 1: loss=0.0245, val_loss=0.0198
Epoch 2: loss=0.0189, val_loss=0.0156  <- Good (decreasing)
Epoch 3: loss=0.0145, val_loss=0.0142
```

**Early stopping trigger** (after patience of 15):
```
Epoch 45: val_loss stops improving for 15 epochs
>> STOPPING - Restoring model from Epoch 30 (best)
```

**Validation metrics**:
- MSE: Should be < 0.001
- MAE: Should be < 0.015
- MAPE: Should be < 5%

### Common Issues During Training

**Issue 1: Insufficient Data**
```
Output: "Insufficient data for XRPUSDT 15m"
Cause: Cryptocurrency has <100 samples or <1000 total points
Solution: Skip this pair, continue with others
Action: Automatic skip, no intervention needed
```

**Issue 2: Out of Memory**
```
Error: "ResourceExhaustedError: OOM when allocating..."
Cause: Batch size too large for GPU memory
Solution: 
  1. Reduce batch_size from 32 to 16 or 8
  2. Reduce sequence_length from 60 to 40 or 30
Edit: train_v7_main.py line 270: batch_size=16
```

**Issue 3: Data Fetch Failure**
```
Output: "Binance fetch error: Connection timeout"
Cause: API rate limit or network issue
Automatic: Falls back to Yahoo Finance
Result: Training continues normally
```

### GPU Utilization

Monitor GPU usage in separate cell:
```python
!nvidia-smi -l 1
```

Expected during training:
- GPU Memory: 12-14 GB used
- GPU Utilization: 90-100%
- Temperature: 60-70°C

---

## After Training Completes

### Verify Upload Success

Check Hugging Face repository:
```
https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/models_v7
```

You should see:
```
models_v7/
  ADAUSDT_15m_v7.h5
  ADAUSDT_1h_v7.h5
  ...
  metadata_v7.json
```

### Download Models Locally (Optional)

```python
from huggingface_hub import snapshot_download

models_path = snapshot_download(
    repo_id='zongowo111/cpb-models',
    repo_type='dataset',
    allow_patterns='models_v7/*'
)
print(f'Models downloaded to: {models_path}')
```

---

## Troubleshooting

### Colab Session Timeout

**Problem**: Running out of Colab's 12-hour session limit

**Solution**:
1. Split training: Train first 10 cryptos, upload
2. Start new session, train remaining 10
3. Or: Increase model training efficiency

**How to resume**:
```python
# In new session
specific_cryptos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
pipeline = TrainingPipeline(output_dir='/content/all_models')
for symbol in specific_cryptos:
    for timeframe in ['15m', '1h']:
        pipeline.train_single_model(symbol, timeframe)
```

### Hugging Face Token Invalid

**Error**: `HTTPError: 401 Client Error: Unauthorized`

**Solution**:
1. Generate new token at https://huggingface.co/settings/tokens
2. Ensure token has "Write" permission
3. Re-run upload cell with new token

### Model File Corrupted

**Error**: Model loads but produces NaN predictions

**Solution**:
1. Retrain affected cryptocurrency
2. Or download from Hugging Face and reload

---

## Performance Benchmarks

### Expected Training Times (T4 GPU)

```
Per cryptocurrency (15m + 1h):
  Data fetch: 10-20 seconds
  Indicator calculation: 5-10 seconds
  Model training (100 epochs): 2-3 minutes
  Total per pair: 3-5 minutes

All 20 cryptocurrencies:
  Total time: 1.5-2.5 hours
```

### Model Sizes

```
Per model (.h5 file): 4.2-4.5 MB
Metadata (JSON): 0.3-0.4 MB
Total (40 models): 165-180 MB
```

---

## Next Steps After Training

1. **Model Evaluation**
   - Download models from Hugging Face
   - Test on recent unseen data
   - Compare with V6 baseline

2. **Backtesting**
   - Use predictions in trading strategy
   - Measure prediction accuracy
   - Calculate Sharpe ratio improvements

3. **Deployment**
   - Load models for real-time prediction
   - Integrate with trading bot
   - Monitor performance metrics

4. **Refinement**
   - Monthly retraining with new data
   - Hyperparameter tuning
   - Feature engineering improvements

---

## Quick Reference Commands

```python
Check GPU
!nvidia-smi

List all models
!ls -lh /content/all_models/

View metadata
!cat /content/all_models/metadata_v7.json

Load model for testing
import tensorflow as tf
model = tf.keras.models.load_model('/content/all_models/BTCUSDT_15m_v7.h5')

Make prediction
predictions = model.predict(X_test)
print('Open, Close, High, Low predictions:', predictions)
```

---

## Support

If issues occur:
1. Check this guide's troubleshooting section
2. Review repo README.md
3. Check MODEL_STRATEGY.md for technical details
4. Open GitHub issue with error details

**Repository**: https://github.com/caizongxun/crypto-v7-models
