import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Concatenate
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
    print('talib not available, using pandas calculations')

print('TensorFlow version:', tf.__version__)
print('GPU Available:', tf.config.list_physical_devices('GPU'))

class CryptoDataFetcher:
    """Fetch cryptocurrency data from multiple sources"""
    
    def __init__(self):
        self.binance_us_client = None
        if HAS_BINANCE:
            try:
                self.binance_us_client = BinanceClient(
                    tld='us',
                    requests_params={"timeout": 10}
                )
            except Exception as e:
                print(f'Binance US client failed: {str(e)[:50]}')
    
    def fetch_from_binance_us(self, symbol, interval, limit=1000):
        """Fetch from Binance US"""
        if self.binance_us_client is None:
            return None
        
        try:
            klines = self.binance_us_client.get_historical_klines(
                symbol, interval, limit=min(limit, 1000)
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[[
                'open', 'high', 'low', 'close', 'volume'
            ]].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            return None
    
    def fetch_from_yfinance(self, symbol, interval, period):
        """Fetch from Yahoo Finance"""
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
            
            if df.empty:
                return None
            
            df.reset_index(inplace=True)
            
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'timestamp'}, inplace=True)
            elif 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
            
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            return None
    
    def get_data(self, symbol, timeframe, limit=10000):
        """Unified interface to fetch data"""
        if self.binance_us_client:
            binance_interval = {'15m': '15m', '1h': '1h'}
            if timeframe in binance_interval:
                try:
                    df = self.fetch_from_binance_us(symbol, timeframe, limit)
                    if df is not None and len(df) > 100:
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
            return df
        
        return None

class TechnicalIndicators:
    """Calculate technical indicators for volatility prediction"""
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Average True Range - Fixed version"""
        try:
            high = np.array(high).flatten()
            low = np.array(low).flatten()
            close = np.array(close).flatten()
            
            if HAS_TALIB:
                try:
                    return talib.ATR(high, low, close, timeperiod=period)
                except:
                    pass
        except:
            pass
        
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(close, period=20, num_std=2):
        """Bollinger Bands with upper, middle, lower bands"""
        close = np.array(close).flatten()
        
        if HAS_TALIB:
            try:
                sma = talib.SMA(close, timeperiod=period)
            except:
                sma = pd.Series(close).rolling(window=period).mean().values
        else:
            sma = pd.Series(close).rolling(window=period).mean().values
        
        std = pd.Series(close).rolling(window=period).std().values
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        bandwidth = (upper - lower) / (sma + 1e-8)
        
        return upper, sma, lower, bandwidth
    
    @staticmethod
    def calculate_rsi(close, period=14):
        """Relative Strength Index"""
        close = np.array(close).flatten()
        
        if HAS_TALIB:
            try:
                return talib.RSI(close, timeperiod=period)
            except:
                pass
        
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def calculate_macd(close):
        """MACD indicator"""
        close = np.array(close).flatten()
        
        if HAS_TALIB:
            try:
                macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                return macd, signal, hist
            except:
                pass
        
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9).mean().values
        hist = macd - signal
        return macd, signal, hist
    
    @staticmethod
    def calculate_volatility(close, period=20):
        """Historical volatility"""
        close = np.array(close).flatten()
        log_returns = np.log(close / np.concatenate([[close[0]], close[:-1]]))
        return pd.Series(log_returns).rolling(window=period).std().values
    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        
        df['atr'] = TechnicalIndicators.calculate_atr(high, low, close)
        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = \
            TechnicalIndicators.calculate_bollinger_bands(close)
        df['rsi'] = TechnicalIndicators.calculate_rsi(close)
        df['macd'], df['macd_signal'], df['macd_hist'] = \
            TechnicalIndicators.calculate_macd(close)
        df['volatility'] = TechnicalIndicators.calculate_volatility(close)
        
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        return df

class DataPreprocessor:
    """Preprocess data for model training"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scalers = {}
    
    def preprocess(self, df):
        """Complete preprocessing pipeline"""
        df = df.copy()
        df = df.dropna()
        
        feature_cols = ['close', 'open', 'high', 'low', 'volume',
                       'atr', 'bb_upper', 'bb_lower', 'bb_width',
                       'rsi', 'macd', 'volatility', 'bb_position']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[available_cols])
        
        self.scalers['scaler'] = scaler
        self.scalers['columns'] = available_cols
        
        return scaled_data, available_cols
    
    def create_sequences(self, data, sequence_length=None):
        """Create sequences for LSTM"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length, :4])
        
        return np.array(X), np.array(y)

class CryptoV7Model:
    """Hybrid BiLSTM model for improved cryptocurrency price prediction"""
    
    def __init__(self, sequence_length=60, features_dim=13):
        self.sequence_length = sequence_length
        self.features_dim = features_dim
        self.model = None
    
    def build_model(self):
        """Build hybrid BiLSTM architecture"""
        input_layer = Input(shape=(self.sequence_length, self.features_dim))
        
        bi_lstm1 = Bidirectional(LSTM(128, return_sequences=True,
                                     activation='relu'))(input_layer)
        dropout1 = Dropout(0.2)(bi_lstm1)
        
        bi_lstm2 = Bidirectional(LSTM(64, return_sequences=True,
                                     activation='relu'))(dropout1)
        dropout2 = Dropout(0.2)(bi_lstm2)
        
        bi_lstm3 = Bidirectional(LSTM(32, return_sequences=False,
                                     activation='relu'))(dropout2)
        dropout3 = Dropout(0.2)(bi_lstm3)
        
        dense1 = Dense(64, activation='relu')(dropout3)
        dense2 = Dense(32, activation='relu')(dense1)
        
        open_output = Dense(1, name='open')(dense2)
        close_output = Dense(1, name='close')(dense2)
        high_output = Dense(1, name='high')(dense2)
        low_output = Dense(1, name='low')(dense2)
        
        self.model = Model(inputs=input_layer,
                          outputs=[open_output, close_output, high_output, low_output])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['mse', 'mse', 'mse', 'mse'],
            loss_weights=[1.0, 1.0, 0.8, 0.8],
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        y_train_list = [y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3]]
        y_val_list = [y_val[:, 0], y_val[:, 1], y_val[:, 2], y_val[:, 3]]
        
        history = self.model.fit(
            X_train, y_train_list,
            validation_data=(X_val, y_val_list),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        outputs = self.model.predict(X, verbose=0)
        predictions = np.column_stack(outputs)
        return predictions
    
    def save(self, filepath):
        """Save model"""
        if self.model:
            self.model.save(filepath)

class TrainingPipeline:
    """Complete training pipeline for all cryptocurrencies"""
    
    CRYPTO_PAIRS = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'BNB': 'BNBUSDT',
        'XRP': 'XRPUSDT',
        'ADA': 'ADAUSDT',
        'DOGE': 'DOGEUSDT',
        'SOL': 'SOLUSDT',
        'LINK': 'LINKUSDT',
        'MATIC': 'MATICUSDT',
        'AVAX': 'AVAXUSDT',
        'UNI': 'UNIUSDT',
        'LTC': 'LTCUSDT',
        'BCH': 'BCHUSDT',
        'ETC': 'ETCUSDT',
        'XLM': 'XLMUSDT',
        'VET': 'VETUSDT',
        'FIL': 'FILUSDT',
        'THETA': 'THETAUSDT',
        'NEAR': 'NEARUSDT',
        'APE': 'APEUSDT'
    }
    
    TIMEFRAMES = ['15m', '1h']
    
    def __init__(self, output_dir='/content/all_models'):
        self.output_dir = output_dir
        self.fetcher = CryptoDataFetcher()
        self.models_metadata = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def train_single_model(self, symbol, timeframe, limit=10000):
        """Train model for single cryptocurrency and timeframe"""
        print(f'Training {symbol} {timeframe}...', end=' ', flush=True)
        
        df = self.fetcher.get_data(symbol, timeframe, limit)
        if df is None or len(df) < 100:
            print('No data')
            return False
        
        df = TechnicalIndicators.add_all_indicators(df)
        df = df.dropna()
        
        if len(df) < 100:
            print('Insufficient data')
            return False
        
        preprocessor = DataPreprocessor(sequence_length=60)
        scaled_data, feature_cols = preprocessor.preprocess(df)
        X, y = preprocessor.create_sequences(scaled_data)
        
        if len(X) < 100:
            print('Insufficient sequences')
            return False
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = CryptoV7Model(sequence_length=60, features_dim=len(feature_cols))
        model.build_model()
        
        history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
        
        model_path = os.path.join(self.output_dir, f'{symbol}_{timeframe}_v7.h5')
        model.save(model_path)
        
        y_pred = model.predict(X_val)
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
            'timestamp': datetime.now().isoformat()
        }
        
        self.models_metadata[f'{symbol}_{timeframe}'] = metadata
        print(f'MAPE: {mape:.2f}%')
        
        return True
    
    def train_all_models(self):
        """Train models for all cryptocurrencies and timeframes"""
        success_count = 0
        total_count = len(self.CRYPTO_PAIRS) * len(self.TIMEFRAMES)
        
        for crypto, symbol in self.CRYPTO_PAIRS.items():
            for timeframe in self.TIMEFRAMES:
                try:
                    if self.train_single_model(symbol, timeframe):
                        success_count += 1
                except Exception as e:
                    print(f'Error: {str(e)[:40]}')
        
        metadata_path = os.path.join(self.output_dir, 'metadata_v7.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)
        
        print(f'\nTraining completed: {success_count}/{total_count} models')
        return self.models_metadata

if __name__ == '__main__':
    pipeline = TrainingPipeline(output_dir='/content/all_models')
    pipeline.train_all_models()
