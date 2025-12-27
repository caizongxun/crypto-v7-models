import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging
import traceback
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

try:
    from binance.client import Client as BinanceClient
    HAS_BINANCE = True
except:
    print("Warning: python-binance not installed. Install with: pip install python-binance")
    HAS_BINANCE = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

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

class CryptoDataFetcher:
    def __init__(self, data_dir='/content/klines_data'):
        self.binance_us_client = None
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        if HAS_BINANCE:
            try:
                self.binance_us_client = BinanceClient(tld='us', requests_params={"timeout": 30})
                print("✓ Binance US client initialized")
            except Exception as e:
                print(f"Warning: Could not initialize Binance US client: {e}")
    
    def fetch_from_binance_us(self, symbol, interval, limit=10000):
        if self.binance_us_client is None:
            return None
        try:
            # Binance API 最多每次取 1000 根
            all_klines = []
            limit_per_request = 1000
            requests_count = (limit + limit_per_request - 1) // limit_per_request
            
            for i in range(requests_count):
                start_time = None
                if all_klines:
                    start_time = all_klines[-1][6] + 1  # use close_time + 1ms of last kline
                
                klines = self.binance_us_client.get_historical_klines(
                    symbol, interval, limit=limit_per_request, startTime=start_time
                )
                if not klines:
                    break
                all_klines.extend(klines)
                
                if len(all_klines) >= limit:
                    all_klines = all_klines[:limit]
                    break
            
            if not all_klines:
                return None
            
            df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"  Binance error: {str(e)[:50]}")
            return None
    
    def get_data(self, symbol, timeframe, limit=10000):
        if self.binance_us_client:
            binance_interval = {'15m': '15m', '1h': '1h'}
            if timeframe in binance_interval:
                try:
                    df = self.fetch_from_binance_us(symbol, timeframe, limit)
                    if df is not None and len(df) > 100:
                        self.save_klines(symbol, timeframe, df, source='binance_us')
                        return df
                except:
                    pass
        return None
    
    def save_klines(self, symbol, timeframe, df, source='unknown'):
        filename = os.path.join(self.data_dir, f'{symbol}_{timeframe}_{source}.csv')
        df.to_csv(filename, index=False)
        print(f'  → Saved {len(df)} klines to {filename}')

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
    def calculate_roc(close, period=12):
        close = np.array(close).flatten()
        roc = (close - np.concatenate([[close[0]]*period, close[:-period]])) / np.concatenate([[close[0]]*period, close[:-period]]) * 100
        return roc
    
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
        df['roc'] = TechnicalIndicators.calculate_roc(close)
        
        # 修復 bb_position
        bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position'] = bb_position.clip(0, 1).values
        
        # 正常化 volume
        df['volume_norm'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-8)
        
        return df

class DataPreprocessor:
    def __init__(self, sequence_length=120):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def preprocess(self, df):
        df = df.copy()
        df = df.dropna()
        
        feature_cols = ['close', 'open', 'high', 'low', 'volume_norm', 'atr', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'macd', 'volatility', 'bb_position', 'roc']
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

class CryptoV7OptimizedModel:
    """
    改進的加密貨幣模組，基於最新的研究
    
    改進：
    1. 更深的 Bidirectional LSTM 層數（4層）
    2. Layer Normalization 穩定訓練
    3. 更澀的正則化
    4. 更好的紁率配置
    5. 優化的你好度函數
    """
    
    def __init__(self, sequence_length=120, features_dim=14):
        self.sequence_length = sequence_length
        self.features_dim = features_dim
        self.model = None
    
    def build_model(self):
        input_layer = Input(shape=(self.sequence_length, self.features_dim))
        
        # 第 1 層 Bidirectional LSTM
        bi_lstm1 = Bidirectional(LSTM(256, return_sequences=True, activation='relu', 
                                      recurrent_regularizer=l1_l2(1e-5, 1e-5)))(input_layer)
        ln1 = LayerNormalization()(bi_lstm1)
        dropout1 = Dropout(0.3)(ln1)
        
        # 第 2 層 Bidirectional LSTM
        bi_lstm2 = Bidirectional(LSTM(128, return_sequences=True, activation='relu',
                                      recurrent_regularizer=l1_l2(1e-5, 1e-5)))(dropout1)
        ln2 = LayerNormalization()(bi_lstm2)
        dropout2 = Dropout(0.3)(ln2)
        
        # 第 3 層 Bidirectional LSTM
        bi_lstm3 = Bidirectional(LSTM(64, return_sequences=True, activation='relu',
                                      recurrent_regularizer=l1_l2(1e-5, 1e-5)))(dropout2)
        ln3 = LayerNormalization()(bi_lstm3)
        dropout3 = Dropout(0.2)(ln3)
        
        # 第 4 層 Bidirectional LSTM (return_sequences=False)
        bi_lstm4 = Bidirectional(LSTM(32, return_sequences=False, activation='relu',
                                      recurrent_regularizer=l1_l2(1e-5, 1e-5)))(dropout3)
        ln4 = LayerNormalization()(bi_lstm4)
        dropout4 = Dropout(0.2)(ln4)
        
        # Dense 層
        dense1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5))(dropout4)
        dropout5 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5))(dropout5)
        dropout6 = Dropout(0.1)(dense2)
        dense3 = Dense(32, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5))(dropout6)
        
        # 輸出層（三個絶對值輸出）
        open_output = Dense(1, name='open')(dense3)
        close_output = Dense(1, name='close')(dense3)
        high_output = Dense(1, name='high')(dense3)
        low_output = Dense(1, name='low')(dense3)
        
        self.model = Model(inputs=input_layer, outputs=[open_output, close_output, high_output, low_output])
        
        # 改進的 optimizer
        optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss={'open': 'mse', 'close': 'mse', 'high': 'mse', 'low': 'mse'},
            loss_weights={'open': 0.9, 'close': 1.2, 'high': 0.9, 'low': 0.9},
            metrics={'open': 'mae', 'close': 'mae', 'high': 'mae', 'low': 'mae'}
        )
        
        return self.model
    
    def train(self, X_train, y_train_open, y_train_close, y_train_high, y_train_low, 
              X_val, y_val_open, y_val_close, y_val_high, y_val_low, epochs=200, batch_size=16):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
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

class OptimizedTrainingPipeline:
    CRYPTO_PAIRS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'DOGE': 'DOGEUSDT', 'SOL': 'SOLUSDT', 'LINK': 'LINKUSDT', 'MATIC': 'MATICUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT', 'LTC': 'LTCUSDT', 'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'XLM': 'XLMUSDT', 'VET': 'VETUSDT', 'FIL': 'FILUSDT', 'THETA': 'THETAUSDT', 'NEAR': 'NEARUSDT', 'APE': 'APEUSDT'}
    TIMEFRAMES = ['15m', '1h']
    
    def __init__(self, output_dir='/content/all_models', klines_dir='/content/klines_data', klines_limit=8000):
        self.output_dir = output_dir
        self.klines_dir = klines_dir
        self.klines_limit = klines_limit  # 7000-10000 根
        self.fetcher = CryptoDataFetcher(data_dir=klines_dir)
        self.models_metadata = {}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(klines_dir, exist_ok=True)
    
    def train_single_model(self, symbol, timeframe):
        print(f'\nTraining {symbol} {timeframe}...')
        print(f'  ← Fetching {self.klines_limit} klines...', end=' ', flush=True)
        df = self.fetcher.get_data(symbol, timeframe, limit=self.klines_limit)
        if df is None or len(df) < 500:
            print('✗ No data')
            return False
        print(f'✓ Got {len(df)} klines')
        
        print(f'  ← Adding indicators...', end=' ', flush=True)
        df = TechnicalIndicators.add_all_indicators(df)
        df = df.dropna()
        if len(df) < 500:
            print('✗ Insufficient data')
            return False
        print(f'✓ {len(df)} rows after cleanup')
        
        print(f'  ← Building sequences...', end=' ', flush=True)
        preprocessor = DataPreprocessor(sequence_length=120)
        scaled_data, feature_cols = preprocessor.preprocess(df)
        X, y_open, y_close, y_high, y_low = preprocessor.create_sequences(scaled_data)
        if len(X) < 200:
            print('✗ Insufficient sequences')
            return False
        print(f'✓ {len(X)} sequences')
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_open_train, y_open_val = y_open[:split_idx], y_open[split_idx:]
        y_close_train, y_close_val = y_close[:split_idx], y_close[split_idx:]
        y_high_train, y_high_val = y_high[:split_idx], y_high[split_idx:]
        y_low_train, y_low_val = y_low[:split_idx], y_low[split_idx:]
        
        print(f'  ← Training model (seq=120, features={len(feature_cols)})...', end=' ', flush=True)
        model = CryptoV7OptimizedModel(sequence_length=120, features_dim=len(feature_cols))
        model.build_model()
        history = model.train(
            X_train, y_open_train, y_close_train, y_high_train, y_low_train,
            X_val, y_open_val, y_close_val, y_high_val, y_low_val,
            epochs=200, batch_size=16
        )
        print('✓ Training complete')
        
        print(f'  ← Saving model...', end=' ', flush=True)
        model_path = os.path.join(self.output_dir, f'{symbol}_{timeframe}_v7_opt.keras')
        model.save(model_path)
        print(f'✓ Saved')
        
        print(f'  ← Evaluating...', end=' ', flush=True)
        y_pred = model.predict(X_val)
        y_val = np.column_stack([y_open_val, y_close_val, y_high_val, y_low_val])
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f'✓ MAPE: {mape:.2f}%')
        
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'sequence_length': 120,
            'klines_count': len(df),
            'features': feature_cols,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'val_mse': float(mse),
            'val_mae': float(mae),
            'val_mape': float(mape),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'V7_Optimized'
        }
        self.models_metadata[f'{symbol}_{timeframe}'] = metadata
        return True
    
    def train_all_models(self):
        success_count = 0
        total_count = len(self.CRYPTO_PAIRS) * len(self.TIMEFRAMES)
        for crypto, symbol in self.CRYPTO_PAIRS.items():
            for timeframe in self.TIMEFRAMES:
                try:
                    if self.train_single_model(symbol, timeframe):
                        success_count += 1
                except Exception as e:
                    print(f'✗ Error: {str(e)[:80]}')
        metadata_path = os.path.join(self.output_dir, 'metadata_v7_opt.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
        print(f'\n\n✓ Training completed: {success_count}/{total_count} models')
        print(f'✓ Klines saved in: {self.klines_dir}')
        print(f'✓ Models saved in: {self.output_dir}')
        return self.models_metadata

if __name__ == '__main__':
    pipeline = OptimizedTrainingPipeline(
        output_dir='/content/all_models', 
        klines_dir='/content/klines_data',
        klines_limit=8000  # 使用 8000 根 K 棒（2023-2024 數個月的數據）
    )
    pipeline.train_all_models()
