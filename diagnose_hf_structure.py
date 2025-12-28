#!/usr/bin/env python
"""
診斷不同的下載方法以找到實際的 klines 檔案
"""

import requests
import json
from io import StringIO
import pandas as pd

print("=" * 80)
print("HF 數據集結構診斷")
print("=" * 80)

HF_REPO = 'zongowo111/cpb-models'
BASE_URL = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main'

# 方法 1: 下載 klines_summary_binance_us.json 來了解數據結構
print("\n[1] 嘗試下載 klines_summary_binance_us.json...")
try:
    url = f'{BASE_URL}/klines_summary_binance_us.json'
    print(f"URL: {url}")
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n數據結構模式:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])  # 只顯示前 2000 字符
        
        # 取一個第一的幣種數據
        first_symbol = list(data['summary'].keys())[0]
        print(f"\n第一個幣種: {first_symbol}")
        print(json.dumps(data['summary'][first_symbol], indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")

# 方法 2: 直接嘗試下載 CSV 檔案
print("\n" + "=" * 80)
print("[2] 嘗試下載具體 CSV 檔案...")

test_symbols = ['BTCUSDT', 'ETHUSDT']
test_timeframes = ['15m', '1h']

for symbol in test_symbols:
    for tf in test_timeframes:
        # 嘉推 1: {symbol}_{tf}_binance_us.csv
        filename1 = f'{symbol}_{tf}_binance_us.csv'
        url1 = f'{BASE_URL}/klines_binance_us/{filename1}'
        
        # 嘉推 2: 粗文件名
        filename2 = f'{symbol}_{tf}.csv'
        url2 = f'{BASE_URL}/klines_binance_us/{filename2}'
        
        # 嘉推 3: 根目錄
        filename3 = f'{symbol}_{tf}.csv'
        url3 = f'{BASE_URL}/{filename3}'
        
        # 嘉推 4: 大寫文件名
        filename4 = f'{symbol.upper()}_{tf.upper()}.csv'
        url4 = f'{BASE_URL}/{filename4}'
        
        for url, fname in [(url1, filename1), (url2, filename2), (url3, filename3), (url4, filename4)]:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  ✓ 找到: {url}")
                    # 芽瞧內容
                    df = pd.read_csv(StringIO(response.text))
                    print(f"    欄位: {list(df.columns)}")
                    print(f"    行數: {len(df)}")
                    print(f"    第一行: {df.iloc[0].to_dict()}")
                    break
                elif response.status_code == 404:
                    print(f"  ✗ 404 Not Found: {url}")
            except Exception as e:
                print(f"  ✗ Error ({fname}): {str(e)[:50]}")

# 方法 3: 列出所有 JSON 檔案
print("\n" + "=" * 80)
print("[3] 列出所有 JSON 檔案符合模式...")
for symbol in test_symbols:
    for tf in test_timeframes:
        filename = f'{symbol}_{tf}_metrics.json'
        url = f'{BASE_URL}/{filename}'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"  ✓ 找到 metrics: {filename}")
            data = response.json()
            # 取 csv_path 來了解實際位置
            if 'csv_path' in data:
                print(f"    CSV Path: {data['csv_path']}")

print("\n" + "=" * 80)
print("診斷完成")
