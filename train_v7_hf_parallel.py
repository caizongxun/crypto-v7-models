import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging
import traceback
import concurrent.futures
import threading
from queue import Queue

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import requests
from io import StringIO

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

# GPU 優化設置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'GPU Setup: {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)')
    except RuntimeError as e:
        print(e)
else:
    print('No GPU found. Using CPU.')

print('TensorFlow version:', tf.__version__)
print('GPU Available:', tf.config.list_physical_devices('GPU'))

class CryptoDataFetcherHF:
    """從 Hugging Face 數據集下載 klines 數據"""
    def __init__(self, data_dir='/content/klines_data', hf_repo_id='zongowo111/cpb-models'):
        self.data_dir = data_dir
        self.hf_repo_id = hf_repo_id
        self.hf_url_base = f'https://huggingface.co/datasets/{hf_repo_id}/resolve/main'
        os.makedirs(data_dir, exist_ok=True)
        print("✓ Hugging Face 直接文件加載器已初始化")
    
    def fetch_from_hf_direct(self, symbol, timeframe):
        """
        從 Hugging Face 直接下載 CSV 檔案
        檔案結構: klines_binance_us/{SYMBOL}/{SYMBOL}_{timeframe}_binance_us.csv
        """
        try:
            csv_filename = f'{symbol}_{timeframe}_binance_us.csv'
            url = f'{self.hf_url_base}/klines_binance_us/{symbol}/{csv_filename}'
            
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                return None
            
            df = pd.read_csv(StringIO(response.text))
            
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            if len(available_cols) < 5:
                df.columns = df.columns.str.lower()
                available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 5:
                return None
            
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    if 'time' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['time'])
                    elif 'datetime' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['datetime'])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) < 10:
                return None
            
            return df
            
        except Exception as e:
            return None
    
    def get_data(self, symbol, timeframe, limit=10000):
        df = self.fetch_from_hf_direct(symbol, timeframe)
        
        if df is None or len(df) < 100:
            return None
        
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        
        self.save_klines(symbol, timeframe, df, source='huggingface')
        return df
    
    def save_klines(self, symbol, timeframe, df, source='huggingface'):
        filename = os.path.join(self.data_dir, f'{symbol}_{timeframe}_{source}.csv')
        df.to_csv(filename, index=False)

class TechnicalIndicators:
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        high = np.array(high).flatten()
        low = np.array(low).flatten()
        close = np.array(close).flatten()
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return pd.Series(tr).rolling(window=period).mean().values
    
    @staticmethod
    def calculate_bollinger_bands(close, period=20, num_std=2):
        close = np.array(close).flatten()
        sma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        bandwidth = (upper - lower) / (sma + 1e-8)
        return upper, sma, lower, bandwidth
    
    @staticmethod
    def calculate_rsi(close, period=14):
        close = np.array(close).flatten()
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def calculate_macd(close):
        close = np.array(close).flatten()
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9).mean().values
        hist = macd - signal
        return macd, signal, hist
    
    @staticmethod
    def calculate_volatility(close, period=20):
        close = np.array(close).flatten()
        log_returns = np.log(close / np.concatenate([[close[0]], close[:-1]]))
        return pd.Series(log_returns).rolling(window=period).std().values
    
    @staticmethod
    def add_all_indicators(df):
        df = df.copy()
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        df['atr'] = TechnicalIndicators.calculate_atr(high, low, close)
        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = TechnicalIndicators.calculate_bollinger_bands(close)
        df['rsi'] = TechnicalIndicators.calculate_rsi(close)
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.calculate_macd(close)
        df['volatility'] = TechnicalIndicators.calculate_volatility(close)
        bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position'] = bb_position.clip(0, 1).values
        return df

class DataPreprocessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def preprocess(self, df):
        df = df.copy()
        df = df.dropna()
        feature_cols = ['close', 'open', 'high', 'low', 'volume', 'atr', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'macd', 'volatility', 'bb_position']
        available_cols = [col for col in feature_cols if col in df.columns]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[available_cols])
        self.scalers['scaler'] = scaler
        self.scalers['columns'] = available_cols
        return scaled_data, available_cols
    
    def create_sequences(self, data, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        X, y_open, y_close, y_high, y_low = [], [], [], [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y_open.append(data[i+sequence_length, 1])
            y_close.append(data[i+sequence_length, 0])
            y_high.append(data[i+sequence_length, 2])
            y_low.append(data[i+sequence_length, 3])
        return np.array(X), np.array(y_open), np.array(y_close), np.array(y_high), np.array(y_low)

class CryptoV7Model:
    def __init__(self, sequence_length=60, features_dim=13):
        self.sequence_length = sequence_length
        self.features_dim = features_dim
        self.model = None
    
    def build_model(self):
        input_layer = Input(shape=(self.sequence_length, self.features_dim))
        
        bi_lstm1 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'))(input_layer)
        dropout1 = Dropout(0.2)(bi_lstm1)
        
        bi_lstm2 = Bidirectional(LSTM(64, return_sequences=True, activation='relu'))(dropout1)
        dropout2 = Dropout(0.2)(bi_lstm2)
        
        bi_lstm3 = Bidirectional(LSTM(32, return_sequences=False, activation='relu'))(dropout2)
        dropout3 = Dropout(0.2)(bi_lstm3)
        
        dense1 = Dense(64, activation='relu')(dropout3)
        dense2 = Dense(32, activation='relu')(dense1)
        
        open_output = Dense(1, name='open')(dense2)
        close_output = Dense(1, name='close')(dense2)
        high_output = Dense(1, name='high')(dense2)
        low_output = Dense(1, name='low')(dense2)
        
        self.model = Model(inputs=input_layer, outputs=[open_output, close_output, high_output, low_output])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'open': 'mse', 'close': 'mse', 'high': 'mse', 'low': 'mse'},
            loss_weights={'open': 1.0, 'close': 1.0, 'high': 0.8, 'low': 0.8},
            metrics={'open': 'mae', 'close': 'mae', 'high': 'mae', 'low': 'mae'}
        )
        
        return self.model
    
    def train(self, X_train, y_train_open, y_train_close, y_train_high, y_train_low, 
              X_val, y_val_open, y_val_close, y_val_high, y_val_low, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        y_train_dict = {
            'open': y_train_open,
            'close': y_train_close,
            'high': y_train_high,
            'low': y_train_low
        }
        
        y_val_dict = {
            'open': y_val_open,
            'close': y_val_close,
            'high': y_val_high,
            'low': y_val_low
        }
        
        history = self.model.fit(
            X_train, y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        outputs = self.model.predict(X, verbose=0)
        predictions = np.column_stack(outputs)
        return predictions
    
    def save(self, filepath):
        if self.model:
            self.model.save(filepath)

class ParallelTrainingPipeline:
    CRYPTO_PAIRS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'DOGE': 'DOGEUSDT', 'SOL': 'SOLUSDT', 'LINK': 'LINKUSDT', 'MATIC': 'MATICUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT', 'LTC': 'LTCUSDT', 'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'XLM': 'XLMUSDT', 'VET': 'VETUSDT', 'FIL': 'FILUSDT', 'THETA': 'THETAUSDT', 'NEAR': 'NEARUSDT', 'APE': 'APEUSDT'}
    TIMEFRAMES = ['15m', '1h']
    
    def __init__(self, output_dir='/content/all_models', klines_dir='/content/klines_data', hf_repo_id='zongowo111/cpb-models', max_workers=2):
        self.output_dir = output_dir
        self.klines_dir = klines_dir
        self.hf_repo_id = hf_repo_id
        self.fetcher = CryptoDataFetcherHF(data_dir=klines_dir, hf_repo_id=hf_repo_id)
        self.models_metadata = {}
        self.max_workers = max_workers
        self.lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(klines_dir, exist_ok=True)
        print(f"✓ 並行訓練管道已初始化 (max_workers={max_workers})")
    
    def train_single_model(self, symbol, timeframe, limit=10000):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Training {symbol} {timeframe}...')
        
        df = self.fetcher.get_data(symbol, timeframe, limit)
        if df is None or len(df) < 100:
            print(f'  ✗ No data for {symbol} {timeframe}')
            return False
        
        df = TechnicalIndicators.add_all_indicators(df)
        df = df.dropna()
        if len(df) < 100:
            print(f'  ✗ Insufficient data for {symbol} {timeframe}')
            return False
        
        preprocessor = DataPreprocessor(sequence_length=60)
        scaled_data, feature_cols = preprocessor.preprocess(df)
        X, y_open, y_close, y_high, y_low = preprocessor.create_sequences(scaled_data)
        if len(X) < 100:
            print(f'  ✗ Insufficient sequences for {symbol} {timeframe}')
            return False
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_open_train, y_open_val = y_open[:split_idx], y_open[split_idx:]
        y_close_train, y_close_val = y_close[:split_idx], y_close[split_idx:]
        y_high_train, y_high_val = y_high[:split_idx], y_high[split_idx:]
        y_low_train, y_low_val = y_low[:split_idx], y_low[split_idx:]
        
        model = CryptoV7Model(sequence_length=60, features_dim=len(feature_cols))
        model.build_model()
        history = model.train(
            X_train, y_open_train, y_close_train, y_high_train, y_low_train,
            X_val, y_open_val, y_close_val, y_high_val, y_low_val,
            epochs=100, batch_size=32
        )
        
        model_path = os.path.join(self.output_dir, f'{symbol}_{timeframe}_v7_hf.keras')
        model.save(model_path)
        
        y_pred = model.predict(X_val)
        y_val = np.column_stack([y_open_val, y_close_val, y_high_val, y_low_val])
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'sequence_length': 60,
            'features': feature_cols,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'val_mse': float(mse),
            'val_mae': float(mae),
            'val_mape': float(mape),
            'model_path': model_path,
            'data_source': 'huggingface',
            'hf_repo': self.hf_repo_id,
            'timestamp': datetime.now().isoformat()
        }
        
        with self.lock:
            self.models_metadata[f'{symbol}_{timeframe}'] = metadata
        
        print(f'  ✓ {symbol} {timeframe} - MAPE: {mape:.2f}%')
        return True
    
    def train_all_models_parallel(self):
        # 創建任務列表
        tasks = []
        for crypto, symbol in self.CRYPTO_PAIRS.items():
            for timeframe in self.TIMEFRAMES:
                tasks.append((symbol, timeframe))
        
        print(f"\n{'='*80}")
        print(f"開始並行訓練 {len(tasks)} 個模型")
        print(f"並行度: {self.max_workers}")
        print(f"{'='*80}\n")
        
        success_count = 0
        failed_tasks = []
        
        # 使用 ThreadPoolExecutor 進行並行訓練
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.train_single_model, symbol, timeframe): (symbol, timeframe) 
                      for symbol, timeframe in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                symbol, timeframe = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failed_tasks.append((symbol, timeframe))
                except Exception as e:
                    print(f'✗ Error for {symbol} {timeframe}: {str(e)[:80]}')
                    failed_tasks.append((symbol, timeframe))
        
        metadata_path = os.path.join(self.output_dir, 'metadata_v7_hf.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"訓練完成")
        print(f"{'='*80}")
        print(f"✓ 成功: {success_count}/{len(tasks)} 模型")
        print(f"✗ 失敗: {len(failed_tasks)} 模型")
        if failed_tasks:
            print(f"失敗列表: {failed_tasks}")
        print(f"\n✓ Klines 已保存到: {self.klines_dir}")
        print(f"✓ Models 已保存到: {self.output_dir}")
        print(f"✓ Metadata 已保存到: {metadata_path}")
        print(f"{'='*80}")
        
        return self.models_metadata

if __name__ == '__main__':
    # 根據 GPU 數量調整 max_workers
    # 1 GPU: max_workers=2-3
    # 2 GPUs: max_workers=4-6
    pipeline = ParallelTrainingPipeline(
        output_dir='/content/all_models',
        klines_dir='/content/klines_data',
        hf_repo_id='zongowo111/cpb-models',
        max_workers=2  # 調整此值
    )
    pipeline.train_all_models_parallel()
