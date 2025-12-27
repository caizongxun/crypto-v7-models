#!/usr/bin/env python3
"""
從 Binance US 下載加密貨幣 K 線資料（純 REST API，無依賴庫）

修正：使用 endTime 參數代替 startTime，避免重複數據
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd
import requests

# 設定
CRYPTO_PAIRS: Dict[str, str] = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT',
    'ADA': 'ADAUSDT', 'DOGE': 'DOGEUSDT', 'SOL': 'SOLUSDT', 'LINK': 'LINKUSDT',
    'MATIC': 'MATICUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT', 'LTC': 'LTCUSDT',
    'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'XLM': 'XLMUSDT', 'VET': 'VETUSDT',
    'FIL': 'FILUSDT', 'THETA': 'THETAUSDT', 'NEAR': 'NEARUSDT', 'APE': 'APEUSDT'
}

TIMEFRAMES: List[str] = ['15m', '1h']
MIN_KLINES = 7000
MAX_KLINES = 10000
KLINES_ROOT = 'klines'

BINANCE_US_BASE_URL = 'https://api.binance.us/api/v3'
BINANCE_MAX_PER_REQUEST = 1000
REQUEST_TIMEOUT = 10


class BinanceUSKlinesDownloader:
    def __init__(self, klines_root: str = KLINES_ROOT):
        self.klines_root = klines_root
        self.base_url = BINANCE_US_BASE_URL
        os.makedirs(self.klines_root, exist_ok=True)
        print("✓ Binance US REST API downloader initialized (pure HTTP, no libraries)")

    def download_klines(self, symbol: str, interval: str, target_count: int) -> Optional[pd.DataFrame]:
        """
        策略：使用 endTime 參數往回拉
        - Batch 0: 取最新的 N 根
        - Batch 1+: 用 endTime 設置為前一批最早的時間 - 1，確保權數ہ改變
        """
        remaining = target_count
        all_klines = []
        batch_idx = 0
        end_time = None  # endTime 用於往後為策

        print(f"  ← Fetching {target_count} klines ({interval}) from Binance US REST API...")

        while remaining > 0:
            limit = min(BINANCE_MAX_PER_REQUEST, remaining)

            try:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit,
                }

                if batch_idx == 0:
                    print(f"    · Batch {batch_idx}: fetching latest {limit} klines...")
                else:
                    params['endTime'] = end_time
                    print(f"    · Batch {batch_idx}: fetching from endTime={end_time}...")

                resp = requests.get(
                    f"{self.base_url}/klines",
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    print("    · No more data returned from Binance")
                    break

                # 確保沒有重複：棧數第一根的時間應該比 end_time 更早
                first_kline_time = int(batch[0][0])
                if batch_idx > 0 and first_kline_time >= end_time:
                    print(f"    · Warning: first kline time {first_kline_time} >= endTime {end_time}, possible duplicate!")

                # 新批次插入前面（保持時間升序）
                all_klines = batch + all_klines
                remaining -= len(batch)

                print(
                    f"    · Batch {batch_idx}: got {len(batch)} klines, "
                    f"total={len(all_klines)}, remaining={max(0, remaining)}"
                )

                # 設置下一批的 endTime：當前批最削的時間 - 1
                if len(batch) > 0:
                    oldest_time = int(batch[0][0])
                    end_time = oldest_time - 1

                batch_idx += 1

                if len(batch) < limit:
                    print("    · Binance returned less than requested, likely reached history boundary")
                    break

                time.sleep(0.2)

            except requests.exceptions.RequestException as e:
                print(f"    · Batch {batch_idx} request error: {str(e)[:80]}")
                if batch_idx == 0:
                    return None
                break
            except Exception as e:
                print(f"    · Batch {batch_idx} error: {str(e)[:80]}")
                if batch_idx == 0:
                    return None
                break

        if not all_klines:
            print("  ✗ No klines downloaded")
            return None

        # 截斷到需要的數量
        if len(all_klines) > target_count:
            all_klines = all_klines[-target_count:]

        # 轉換 DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore',
            ],
        )

        # 數據類型轉換
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 只保留必要欄位
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]

        print(f"  ✓ Downloaded {len(df)} klines for {symbol} {interval}")
        return df

    def save_klines_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        symbol_dir = os.path.join(self.klines_root, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        filename = os.path.join(symbol_dir, f"{symbol}_{interval}_binance_us.csv")
        df.to_csv(filename, index=False)
        print(f"  → Saved to {filename}")
        return filename


def download_all_pairs_klines():
    print("="*70)
    print("=  Crypto V7 - Binance US Klines Dataset Downloader (Pure HTTP)  =")
    print("="*70 + "\n")

    downloader = BinanceUSKlinesDownloader(klines_root=KLINES_ROOT)
    summary = {}

    for base, symbol in CRYPTO_PAIRS.items():
        print("-" * 70)
        print(f"Symbol: {symbol}")
        print("-" * 70)

        summary[symbol] = {}

        for interval in TIMEFRAMES:
            df = downloader.download_klines(symbol, interval, target_count=MAX_KLINES)
            if df is None or len(df) < MIN_KLINES:
                print(
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

    print("\n" + "="*70)
    print("=                    Download Summary                             =")
    print("="*70)
    print(f"Summary JSON: {summary_path}")

    total_configs = sum(len(v) for v in summary.values())
    print(f"Successfully downloaded {total_configs} symbol/timeframe combinations:")
    for symbol, config in summary.items():
        for interval, meta in config.items():
            print(
                f"  ✓ {symbol} {interval}: {meta['rows']} rows "
                f"[{meta['start_time'][:10]} → {meta['end_time'][:10]}]"
            )
    
    if total_configs == 0:
        print("\n✗ No data downloaded. Check network connection.")
    else:
        print(f"\n✓ All klines saved in: {KLINES_ROOT}/")


if __name__ == '__main__':
    download_all_pairs_klines()
