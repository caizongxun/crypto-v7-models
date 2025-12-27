#!/usr/bin/env python3
"""
加密貨幣模型預測可視化

Description:
    此檔案可視化 .keras 模型的預測結果，
    將真實價格和預測價格繪製在同一個圖表上進行對比。

Usage:
    python visualize_predictions.py \
        --model_path /path/to/model.keras \
        --klines_path /path/to/klines.csv \
        --symbol BTCUSDT \
        --output /path/to/output.png

Example:
    python visualize_predictions.py \
        --model_path /content/all_models/BTCUSDT_15m_v7.keras \
        --klines_path /content/klines_data/BTCUSDT_15m_yfinance.csv \
        --symbol BTCUSDT_15m
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


class TechnicalIndicators:
    """技術指標計算"""

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
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        df["atr"] = TechnicalIndicators.calculate_atr(high, low, close)
        df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_width"] = (
            TechnicalIndicators.calculate_bollinger_bands(close)
        )
        df["rsi"] = TechnicalIndicators.calculate_rsi(close)
        df["macd"], df["macd_signal"], df["macd_hist"] = (
            TechnicalIndicators.calculate_macd(close)
        )
        df["volatility"] = TechnicalIndicators.calculate_volatility(close)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-8
        )
        df["bb_position"] = df["bb_position"].clip(0, 1)
        return df


class ModelVisualizer:
    """模型預測可視化"""

    def __init__(self, model_path: str, klines_path: str, sequence_length: int = 60):
        """
        初始化可視化器

        Args:
            model_path (str): 模型檔案路徑
            klines_path (str): K線數據檔案路徑
            sequence_length (int): 序列長度 (default: 60)
        """
        self.model_path = model_path
        self.klines_path = klines_path
        self.sequence_length = sequence_length

        # 載入模型
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")

        # 載入 K 線數據
        print(f"\nLoading klines from {klines_path}...")
        self.df = pd.read_csv(klines_path)
        print(f"✓ Loaded {len(self.df)} klines")

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        預處理數據

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (scaled_data, scaler, feature_cols)
        """
        print(f"\nPreprocessing data...")

        # 添加技術指標
        df = TechnicalIndicators.add_all_indicators(self.df)
        df = df.dropna()

        # 特徵選擇
        feature_cols = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "atr",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "rsi",
            "macd",
            "volatility",
            "bb_position",
        ]
        available_cols = [col for col in feature_cols if col in df.columns]

        # 標準化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[available_cols])

        print(f"✓ Data preprocessed: {scaled_data.shape}")
        print(f"✓ Features used: {available_cols}")

        return scaled_data, scaler, available_cols

    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        建立序列

        Args:
            data (np.ndarray): 縮放後的數據

        Returns:
            np.ndarray: 序列數據
        """
        X = []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
        return np.array(X)

    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        進行預測

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (predictions, actual_prices, timestamps)
        """
        print(f"\nPreparing data for prediction...")

        # 預處理
        scaled_data, scaler, feature_cols = self.preprocess_data()
        X = self.create_sequences(scaled_data)

        print(f"\nMaking predictions...")
        predictions = self.model.predict(X, verbose=0)

        # 反縮放預測值
        # predictions shape: (n_samples, 4) - [open, close, high, low]
        # 我們只關心 close (index 1)
        y_pred_close = predictions[:, 1]  # close output

        # 反縮放到原始範圍
        dummy = np.zeros((len(y_pred_close), len(feature_cols)))
        dummy[:, 0] = y_pred_close  # close 在 index 0
        y_pred_close_actual = scaler.inverse_transform(dummy)[:, 0]

        # 獲取實際價格
        df = TechnicalIndicators.add_all_indicators(self.df)
        df = df.dropna()
        actual_prices = df["close"].values[self.sequence_length :]
        timestamps = pd.to_datetime(df["timestamp"].values[self.sequence_length :])

        print(f"✓ Predictions completed")
        print(f"  Prediction samples: {len(y_pred_close_actual)}")
        print(f"  Actual samples: {len(actual_prices)}")

        return y_pred_close_actual, actual_prices, timestamps

    def calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> dict:
        """
        計算性能指標

        Args:
            predictions (np.ndarray): 預測值
            actuals (np.ndarray): 實際值

        Returns:
            dict: 性能指標
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

        return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    def plot_predictions(
        self,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6),
    ):
        """
        繪製預測結果對比圖

        Args:
            output_path (str): 輸出圖片路徑 (optional)
            title (str): 圖表標題 (optional)
            figsize (Tuple[int, int]): 圖表大小
        """
        print(f"\n{'='*70}")
        print(f"Generating visualization...")
        print(f"{'='*70}\n")

        # 進行預測
        predictions, actuals, timestamps = self.predict()

        # 計算指標
        metrics = self.calculate_metrics(predictions, actuals)

        # 建立圖表
        fig, ax = plt.subplots(figsize=figsize)

        # 繪製實際價格
        ax.plot(
            timestamps,
            actuals,
            label="Actual Price",
            color="#2180a5",
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=3,
        )

        # 繪製預測價格
        ax.plot(
            timestamps,
            predictions,
            label="Predicted Price",
            color="#ff6b6b",
            linewidth=2,
            alpha=0.8,
            marker="s",
            markersize=3,
        )

        # 格式化
        if title is None:
            model_name = Path(self.model_path).stem
            title = f"Price Prediction vs Actual - {model_name}"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Timestamp", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price (USDT)", fontsize=12, fontweight="bold")

        # 日期格式
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45, ha="right")

        # 圖例和網格
        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        # 添加性能指標
        metrics_text = f"MSE: {metrics['MSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nRMSE: {metrics['RMSE']:.4f}\nMAPE: {metrics['MAPE']:.2f}%"
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            family="monospace",
        )

        plt.tight_layout()

        # 儲存或顯示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✓ Figure saved to {output_path}")
        else:
            plt.show()

        # 輸出性能指標
        print(f"\n{'='*70}")
        print(f"Performance Metrics")
        print(f"{'='*70}")
        for key, value in metrics.items():
            if key == "MAPE":
                print(f"{key:6s}: {value:10.2f}%")
            else:
                print(f"{key:6s}: {value:10.6f}")
        print(f"{'='*70}\n")

        return fig, metrics


def main():
    """
    主函式
    """
    parser = argparse.ArgumentParser(
        description="Visualize .keras model predictions vs actual prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize BTCUSDT 15m model
  python visualize_predictions.py \
    --model_path /content/all_models/BTCUSDT_15m_v7.keras \
    --klines_path /content/klines_data/BTCUSDT_15m_yfinance.csv \
    --output /tmp/btc_15m.png

  # Visualize ETHUSDT 1h model with custom title
  python visualize_predictions.py \
    --model_path /content/all_models/ETHUSDT_1h_v7.keras \
    --klines_path /content/klines_data/ETHUSDT_1h_yfinance.csv \
    --title "ETH Price Prediction (1h)" \
    --output /tmp/eth_1h.png
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to .keras model file",
    )
    parser.add_argument(
        "--klines_path",
        type=str,
        required=True,
        help="Path to klines CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (if not specified, will display in notebook)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom chart title",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=60,
        help="Sequence length (default: 60)",
    )

    args = parser.parse_args()

    # 驗證檔案存在
    if not Path(args.model_path).exists():
        print(f"✗ Model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.klines_path).exists():
        print(f"✗ Klines file not found: {args.klines_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # 建立可視化器
        visualizer = ModelVisualizer(
            model_path=args.model_path,
            klines_path=args.klines_path,
            sequence_length=args.sequence_length,
        )

        # 繪製
        fig, metrics = visualizer.plot_predictions(
            output_path=args.output, title=args.title
        )

        print(f"\n✓ Visualization completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
