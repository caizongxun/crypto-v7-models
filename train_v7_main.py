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
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import yfinance as yf

try:
    from binance.client import Client as BinanceClient
    HAS_BINANCE = True
except:
    HAS_BINANCE = False

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

class CryptoDataFetcher:
    def __init__(self, data_dir='/content/klines_data'):
        self.binance_us_client = None
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        if HAS_BINANCE:
            try:
                self.binance_us_client = BinanceClient(tld='us', requests_params={"timeout": 10})
            except:
                pass
    
    def fetch_from_binance_us(self, symbol, interval, limit=1000):
        if self.binance_us_client is None:
            return None
        try:
            klines = self.binance_us_client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
            if not klines:
                return None
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except:
            return None
    
    def fetch_from_yfinance(self, symbol, interval, period):
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
            if df.empty:
                return None
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'timestamp'}, inplace=True)
            elif 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
            df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except:
            return None
    
    def get_data(self, symbol, timeframe, limit=10000):
        if self.binance_us_client:
            binance_interval = {'15m': '15m', '1h': '1h'}
            if timeframe in binance_interval:
                try:
                    df = self.fetch_from_binance_us(symbol, timeframe, limit)
                    if df is not None and len(df) > 100:
                        self.save_klines(symbol, timeframe, df, source='binance')
                        return df
                except:
                    pass
        
        yf_interval = {'15m': '15m', '1h': '1h'}
        yf_symbol = symbol.replace('USDT', '') if 'USDT' in symbol else symbol
        if timeframe in yf_interval:
            if timeframe == '15m':
                df = self.fetch_from_yfinance(yf_symbol, yf_interval[timeframe], '60d')
            else:
                df = self.fetch_from_yfinance(yf_symbol, yf_interval[timeframe], '180d')
            if df is not None and len(df) > limit:
                df = df.iloc[-limit:].reset_index(drop=True)
            if df is not None:
                self.save_klines(symbol, timeframe, df, source='yfinance')
            return df
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
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position'] = df['bb_position'].clip(0, 1)
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

class TrainingPipeline:
    # 修改：只訓練 5 個幣種
    CRYPTO_PAIRS = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'BNB': 'BNBUSDT',
        'XRP': 'XRPUSDT',
        'ADA': 'ADAUSDT'
    }
    TIMEFRAMES = ['15m', '1h']
    
    def __init__(self, output_dir='/content/all_models', klines_dir='/content/klines_data'):
        self.output_dir = output_dir
        self.klines_dir = klines_dir
        self.fetcher = CryptoDataFetcher(data_dir=klines_dir)
        self.models_metadata = {}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(klines_dir, exist_ok=True)
    
    def train_single_model(self, symbol, timeframe, limit=10000):
        print(f'\nTraining {symbol} {timeframe}...')
        print(f'  ← Fetching klines...', end=' ', flush=True)
        df = self.fetcher.get_data(symbol, timeframe, limit)
        if df is None or len(df) < 100:
            print('✗ No data')
            return False
        print(f'✓ Got {len(df)} klines')
        
        print(f'  ← Adding indicators...', end=' ', flush=True)
        df = TechnicalIndicators.add_all_indicators(df)
        df = df.dropna()
        if len(df) < 100:
            print('✗ Insufficient data')
            return False
        print(f'✓ {len(df)} rows after cleanup')
        
        print(f'  ← Building sequences...', end=' ', flush=True)
        preprocessor = DataPreprocessor(sequence_length=60)
        scaled_data, feature_cols = preprocessor.preprocess(df)
        X, y_open, y_close, y_high, y_low = preprocessor.create_sequences(scaled_data)
        if len(X) < 100:
            print('✗ Insufficient sequences')
            return False
        print(f'✓ {len(X)} sequences')
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_open_train, y_open_val = y_open[:split_idx], y_open[split_idx:]
        y_close_train, y_close_val = y_close[:split_idx], y_close[split_idx:]
        y_high_train, y_high_val = y_high[:split_idx], y_high[split_idx:]
        y_low_train, y_low_val = y_low[:split_idx], y_low[split_idx:]
        
        print(f'  ← Training model...', end=' ', flush=True)
        model = CryptoV7Model(sequence_length=60, features_dim=len(feature_cols))
        model.build_model()
        history = model.train(
            X_train, y_open_train, y_close_train, y_high_train, y_low_train,
            X_val, y_open_val, y_close_val, y_high_val, y_low_val,
            epochs=100, batch_size=32
        )
        print('✓ Training complete')
        
        print(f'  ← Saving model...', end=' ', flush=True)
        model_path = os.path.join(self.output_dir, f'{symbol}_{timeframe}_v7.keras')
        model.save(model_path)
        print(f'✓ Saved to {model_path}')
        
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
            'sequence_length': 60,
            'features': feature_cols,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'val_mse': float(mse),
            'val_mae': float(mae),
            'val_mape': float(mape),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
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
        metadata_path = os.path.join(self.output_dir, 'metadata_v7.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
        print(f'\n\n✓ Training completed: {success_count}/{total_count} models')
        print(f'✓ Klines saved in: {self.klines_dir}')
        print(f'✓ Models saved in: {self.output_dir}')
        return self.models_metadata

if __name__ == '__main__':
    pipeline = TrainingPipeline(output_dir='/content/all_models', klines_dir='/content/klines_data')
    pipeline.train_all_models()
