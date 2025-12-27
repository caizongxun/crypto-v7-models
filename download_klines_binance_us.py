#!/usr/bin/env python3
"""
下載並保存加密貨幣 K 線資料（Binance US）

用途：
- 只負責抓資料，不做訓練
- 每個幣種、每個時間框架各抓取 7000-10000 根 K 棒
- 儲存到 repo 根目錄下的 klines/ 資料夾
- 檔名格式： klines/{symbol}/{symbol}_{timeframe}_binance_us.csv

使用方式（Colab）：

    !pip install python-binance
    !python download_klines_binance_us.py

之後可再寫獨立腳本負責上傳到 HF datasets
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    from binance.client import Client as BinanceClient
    HAS_BINANCE = True
except Exception as e:
    print("Warning: python-binance not installed. Install with: pip install python-binance")
    HAS_BINANCE = False

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

# 幣種與時間框架
CRYPTO_PAIRS: Dict[str, str] = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT',
    'ADA': 'ADAUSDT', 'DOGE': 'DOGEUSDT', 'SOL': 'SOLUSDT', 'LINK': 'LINKUSDT',
    'MATIC': 'MATICUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT', 'LTC': 'LTCUSDT',
    'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'XLM': 'XLMUSDT', 'VET': 'VETUSDT',
    'FIL': 'FILUSDT', 'THETA': 'THETAUSDT', 'NEAR': 'NEARUSDT', 'APE': 'APEUSDT'
}

TIMEFRAMES: List[str] = ['15m', '1h']

# 每個交易對要抓的 K 線數量範圍
MIN_KLINES = 7000
MAX_KLINES = 10000

# 儲存資料的根目錄（相對於 repo 根目錄）
KLINES_ROOT = 'klines'

# Binance US 連線設定
BINANCE_REQUEST_TIMEOUT = 30
BINANCE_MAX_PER_REQUEST = 1000


# ---------------------------------------------------------------------------
# 工具函數：Binance US K 線下載
# ---------------------------------------------------------------------------

class BinanceUSKlinesDownloader:
    def __init__(self, klines_root: str = KLINES_ROOT, print_prefix: str = ""):
        self.klines_root = klines_root
        self.print_prefix = print_prefix
        os.makedirs(self.klines_root, exist_ok=True)

        self.client: Optional[BinanceClient] = None
        if HAS_BINANCE:
            try:
                self.client = BinanceClient(tld='us', requests_params={"timeout": BINANCE_REQUEST_TIMEOUT})
                self._log("✓ Binance US client initialized")
            except Exception as e:
                self._log(f"✗ Failed to initialize Binance US client: {e}")
                self.client = None

    def _log(self, msg: str) -> None:
        if self.print_prefix:
            print(f"{self.print_prefix}{msg}")
        else:
            print(msg)

    def _ensure_symbol_dir(self, symbol: str) -> str:
        symbol_dir = os.path.join(self.klines_root, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        return symbol_dir

    def download_klines(self, symbol: str, interval: str, target_count: int) -> Optional[pd.DataFrame]:
        """從 Binance US 下載指定數量 K 線，從最新往回抓。"""
        if self.client is None:
            self._log("✗ Binance US client not available")
            return None

        remaining = target_count
        all_klines = []
        batch_idx = 0

        self._log(f"  ← Fetching {target_count} klines ({interval}) from Binance US...")

        while remaining > 0:
            limit = min(BINANCE_MAX_PER_REQUEST, remaining)

            try:
                if batch_idx == 0:
                    # 第一批：直接取最新
                    batch = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                    )
                else:
                    # 後續批次：用前一輪最舊 kline 的開盤時間當 endTime 往前抓
                    oldest_open_time = all_klines[0][0]  # 第一筆是目前最舊的
                    end_time = int(oldest_open_time) - 1
                    batch = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        endTime=end_time,
                    )

                if not batch:
                    self._log("    · No more data returned from Binance")
                    break

                # 新批次插在前面，保持時間升序
                all_klines = batch + all_klines
                remaining -= len(batch)
                batch_idx += 1

                self._log(
                    f"    · Batch {batch_idx}: got {len(batch)} klines, "
                    f"total={len(all_klines)}, remaining={max(0, remaining)}"
                )

                if len(batch) < limit:
                    self._log("    · Binance returned less than requested, likely reached history boundary")
                    break

            except Exception as e:
                self._log(f"    · Batch {batch_idx} error: {str(e)[:80]}")
                break

        if not all_klines:
            self._log("  ✗ No klines downloaded")
            return None

        # 截斷到需要的數量（最舊在前，最新在後）
        if len(all_klines) > target_count:
            all_klines = all_klines[-target_count:]

        df = pd.DataFrame(
            all_klines,
            columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore',
            ],
        )

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 我們只保留必要欄位，其他在訓練前再重算
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]

        self._log(f"  ✓ Downloaded {len(df)} klines for {symbol} {interval}")
        return df

    def save_klines_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        symbol_dir = self._ensure_symbol_dir(symbol)
        filename = os.path.join(symbol_dir, f"{symbol}_{interval}_binance_us.csv")
        df.to_csv(filename, index=False)
        self._log(f"  ✓ Saved to {filename}")
        return filename


# ---------------------------------------------------------------------------
# 主流程：下載所有幣種 + 時間框架
# ---------------------------------------------------------------------------

def download_all_pairs_klines():
    print("======================================================================")
    print("=        Crypto V7 - Binance US Klines Dataset Downloader           =")
    print("======================================================================\n")

    downloader = BinanceUSKlinesDownloader(klines_root=KLINES_ROOT)

    if downloader.client is None:
        print("✗ Binance US client is not available. Abort.")
        return

    summary = {}

    for base, symbol in CRYPTO_PAIRS.items():
        print("----------------------------------------------------------------------")
        print(f"Symbol: {symbol}")
        print("----------------------------------------------------------------------")

        summary[symbol] = {}

        for interval in TIMEFRAMES:
            df = downloader.download_klines(symbol, interval, target_count=MAX_KLINES)
            if df is None or len(df) < MIN_KLINES:
                downloader._log(
                    f"  ✗ Not enough data for {symbol} {interval} "
                    f"(got {0 if df is None else len(df)}, need >= {MIN_KLINES})"
                )
                continue

            csv_path = downloader.save_klines_csv(symbol, interval, df)
            summary[symbol][interval] = {
                'rows': len(df),
                'csv_path': csv_path,
                'start_time': df['open_time'].iloc[0].isoformat(),
                'end_time': df['open_time'].iloc[-1].isoformat(),
            }

    # 寫 summary json
    summary_path = os.path.join(KLINES_ROOT, 'klines_summary_binance_us.json')
    with open(summary_path, 'w') as f:
        json.dump({'generated_at': datetime.utcnow().isoformat() + 'Z', 'summary': summary}, f, indent=2)

    print("\n======================================================================")
    print("=                         Download Summary                           =")
    print("======================================================================")
    print(f"Summary JSON: {summary_path}")

    total_pairs = sum(1 for _ in summary.keys())
    total_configs = sum(len(v) for v in summary.values())
    print(f"Downloaded klines for {total_configs} symbol/timeframe combinations")
    print("Details:")
    for symbol, config in summary.items():
        for interval, meta in config.items():
            print(
                f"  - {symbol} {interval}: {meta['rows']} rows "
                f"[{meta['start_time']} → {meta['end_time']}]"
            )


if __name__ == '__main__':
    download_all_pairs_klines()
