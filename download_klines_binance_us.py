#!/usr/bin/env python3
"""
從 Binance US 下載加密貨幣 K 線資料

用途：
- 只負責抓資料，不做訓練
- 每個幣種、每個時間框架各抓取 7000-10000 根 K 棒
- 儲存到 repo 根目錄下的 klines/ 資料夾
- 檔名格式：klines/{symbol}/{symbol}_{timeframe}_binance_us.csv

使用方式（Colab）：

    !pip install --upgrade python-binance
    !python download_klines_binance_us.py

之後可再寫獨立腳本負責上傳到 HF datasets
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    from binance.client import Client as BinanceClient
    HAS_BINANCE = True
except Exception as e:
    print("Warning: python-binance not installed. Install with: pip install --upgrade python-binance")
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

# Binance 連線設定
BINANCE_REQUEST_TIMEOUT = 30
BINANCE_MAX_PER_REQUEST = 1000  # Binance API 限制


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
                # 使用 US 交易所，不需要 API Key（公開 K 線資料）
                self.client = BinanceClient(tld='us', requests_params={"timeout": BINANCE_REQUEST_TIMEOUT})
                self._log("✓ Binance US client initialized (public API, no auth needed)")
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
        """
        從 Binance US 下載指定數量 K 線。
        
        策略：
        1. 第一批：抓最新的 N 根
        2. 後續批次：用 startTime 往歷史回拉
        
        Args:
            symbol: 交易對，如 'BTCUSDT'
            interval: 時間框架，如 '15m', '1h'
            target_count: 目標 K 線數量
            
        Returns:
            DataFrame 或 None
        """
        if self.client is None:
            self._log("✗ Binance US client not available")
            return None

        remaining = target_count
        all_klines = []
        batch_idx = 0
        start_time = None  # 用於往回抓的起始時間戳

        self._log(f"  ← Fetching {target_count} klines ({interval}) from Binance US...")

        while remaining > 0:
            limit = min(BINANCE_MAX_PER_REQUEST, remaining)

            try:
                if batch_idx == 0:
                    # 第一批：取最新的（不帶 startTime）
                    self._log(f"    · Batch {batch_idx}: fetching latest {limit} klines...")
                    batch = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                    )
                else:
                    # 後續批次：往歷史回拉
                    self._log(f"    · Batch {batch_idx}: fetching from startTime={start_time}...")
                    batch = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        startTime=start_time,
                    )

                if not batch:
                    self._log("    · No more data returned from Binance")
                    break

                # 新批次插入前面（保持時間升序）
                all_klines = batch + all_klines
                remaining -= len(batch)

                self._log(
                    f"    · Batch {batch_idx}: got {len(batch)} klines, "
                    f"total={len(all_klines)}, remaining={max(0, remaining)}"
                )

                # 設置下一批的 startTime（當前批最老 kline 的開始時間 - 1）
                if len(batch) > 0:
                    oldest_time = int(batch[0][0])
                    start_time = oldest_time - 1

                batch_idx += 1

                # 如果這批少於要求的數量，可能已經到了歷史邊界
                if len(batch) < limit:
                    self._log("    · Binance returned less than requested, likely reached history boundary")
                    break

                # 短暫延遲，避免被 rate limit
                time.sleep(0.1)

            except Exception as e:
                self._log(f"    · Batch {batch_idx} error: {str(e)[:80]}")
                if batch_idx == 0:
                    # 第一批就失敗，直接返回
                    return None
                # 後續批次失敗，用已有資料
                break

        if not all_klines:
            self._log("  ✗ No klines downloaded")
            return None

        # 截斷到需要的數量（最新在後面）
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

        # 只保留必要欄位
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]

        self._log(f"  ✓ Downloaded {len(df)} klines for {symbol} {interval}")
        return df

    def save_klines_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        symbol_dir = self._ensure_symbol_dir(symbol)
        filename = os.path.join(symbol_dir, f"{symbol}_{interval}_binance_us.csv")
        df.to_csv(filename, index=False)
        self._log(f"  → Saved to {filename}")
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
        json.dump(
            {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'summary': summary
            },
            f,
            indent=2
        )

    print("\n======================================================================")
    print("=                         Download Summary                           =")
    print("======================================================================")
    print(f"Summary JSON: {summary_path}")

    total_configs = sum(len(v) for v in summary.values())
    print(f"Successfully downloaded {total_configs} symbol/timeframe combinations:")
    for symbol, config in summary.items():
        for interval, meta in config.items():
            print(
                f"  ✓ {symbol} {interval}: {meta['rows']} rows "
                f"[{meta['start_time']} → {meta['end_time']}]"
            )
    
    if total_configs == 0:
        print("\n✗ No data downloaded. Check Binance connection or network.")
    else:
        print(f"\n✓ All klines saved in: {KLINES_ROOT}/")


if __name__ == '__main__':
    download_all_pairs_klines()
