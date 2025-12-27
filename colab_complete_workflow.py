#!/usr/bin/env python3
"""
完整 Colab 工作流程 - 使用優化版本

Description:
    此腳本包含從 git clone 到優化訓練完成的完整工作流程。
    - 8000 根 K 棒
    - 4 層 Bidirectional LSTM
    - 14 個技術指標
    - 優化損失函數

Usage in Colab:
    !pip install python-binance
    !curl -s https://raw.githubusercontent.com/caizongxun/crypto-v7-models/main/colab_complete_workflow.py | python
"""

import os
import sys
import subprocess
from pathlib import Path


class CoLabWorkflow:
    """
    完整的 Colab 工作流程
    """

    def __init__(self):
        self.repo_url = "https://github.com/caizongxun/crypto-v7-models"
        self.repo_dir = "/content/repo"
        self.models_dir = "/content/all_models"
        self.klines_dir = "/content/klines_data"

    def print_section(self, title: str, level: int = 1):
        """
        輸出分段標題

        Args:
            title (str): 標題文本
            level (int): 標題等級 (1-3)
        """
        symbols = {1: "=", 2: "-", 3: "·"}
        symbol = symbols.get(level, "=")
        width = 70
        padding = (width - len(title) - 2) // 2
        print(f"\n{symbol * width}")
        print(f"{symbol} {title.center(width - 4)} {symbol}")
        print(f"{symbol * width}\n")

    def run_command(self, cmd: str, description: str = ""):
        """
        執行命令並顯示進度

        Args:
            cmd (str): 要執行的命令
            description (str): 命令描述
        """
        if description:
            print(f"→ {description}...")
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"✗ 命令執行失敗: {cmd}")
            return False
        print(f"✓ 完成\n")
        return True

    def step_1_git_clone(self):
        """
        第 1 步: Git Clone
        """
        self.print_section("Step 1: Git Clone Repository", level=1)

        if Path(self.repo_dir).exists():
            print(f"Repository already exists at {self.repo_dir}")
            print(f"Updating repository...")
            self.run_command(f"cd {self.repo_dir} && git pull", "Update repository")
        else:
            self.run_command(
                f"git clone {self.repo_url} {self.repo_dir}",
                "Clone repository",
            )

    def step_2_install_dependencies(self):
        """
        第 2 步: 安裝依賴
        """
        self.print_section("Step 2: Install Dependencies", level=1)

        self.run_command("pip install --upgrade pip", "Upgrade pip")

        packages = [
            "tensorflow>=2.13.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "yfinance>=0.2.0",
            "huggingface_hub>=0.16.0",
            "python-binance>=1.0.17",
        ]

        for package in packages:
            self.run_command(f"pip install {package}", f"Install {package.split('>=')[0]}")

    def step_3_verify_setup(self):
        """
        第 3 步: 驗證環境
        """
        self.print_section("Step 3: Verify Environment Setup", level=1)

        print("Checking Python packages...\n")

        try:
            import tensorflow as tf
            print(f"✓ TensorFlow: {tf.__version__}")
            print(f"  GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
        except ImportError:
            print("✗ TensorFlow not found")

        try:
            import numpy as np
            print(f"✓ NumPy: {np.__version__}")
        except ImportError:
            print("✗ NumPy not found")

        try:
            import pandas as pd
            print(f"✓ Pandas: {pd.__version__}")
        except ImportError:
            print("✗ Pandas not found")

        try:
            import sklearn
            print(f"✓ Scikit-learn: {sklearn.__version__}")
        except ImportError:
            print("✗ Scikit-learn not found")

        try:
            import matplotlib
            print(f"✓ Matplotlib: {matplotlib.__version__}")
        except ImportError:
            print("✗ Matplotlib not found")

        try:
            from binance.client import Client
            print(f"✓ python-binance installed")
        except ImportError:
            print("✗ python-binance not found")

        print()

    def step_4_prepare_directories(self):
        """
        第 4 步: 準備目錄
        """
        self.print_section("Step 4: Prepare Directories", level=1)

        directories = [self.models_dir, self.klines_dir]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory}")

        print()

    def step_5_start_training(self):
        """
        第 5 步: 開始訓練（使用優化版本）
        """
        self.print_section("Step 5: Start Optimized Model Training", level=1)

        print(f"Training Configuration (V7 Optimized):")
        print(f"  • K-lines per coin: 8000 (約8 個月 15m 數據)")
        print(f"  • Model architecture: 4-Layer Bidirectional LSTM")
        print(f"  • Sequence length: 120 steps")
        print(f"  • Technical features: 14 indicators")
        print(f"  • Total cryptos: 20")
        print(f"  • Total timeframes: 2 (15m, 1h)")
        print(f"  • Total models to train: 40")
        print(f"  • Expected MAPE: 3-5% (vs 6-8% baseline)")
        print()
        print(f"Models directory: {self.models_dir}")
        print(f"Klines directory: {self.klines_dir}")
        print()
        print(f"Starting training...\n")

        # 直接執行訓練
        try:
            cmd = f"cd {self.repo_dir} && python train_v7_optimized.py"
            result = subprocess.run(cmd, shell=True)
            
            if result.returncode == 0:
                print(f"\n✓ Training completed successfully!")
                return True
            else:
                print(f"\n✗ Training failed with return code {result.returncode}")
                return False

        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_6_summary(self):
        """
        第 6 步: 總結
        """
        self.print_section("Step 6: Summary & Results", level=1)

        print(f"✓ Workflow completed!\n")

        # 檢查生成的檔案
        models_count = len(list(Path(self.models_dir).glob("*.keras"))) if Path(self.models_dir).exists() else 0
        klines_count = len(list(Path(self.klines_dir).glob("*.csv"))) if Path(self.klines_dir).exists() else 0

        print(f"Generated Files:")
        print(f"  • Models: {self.models_dir}")
        print(f"    - {models_count} .keras files")
        print(f"    - metadata_v7_opt.json")
        print(f"  • Klines: {self.klines_dir}")
        print(f"    - {klines_count} .csv files")
        print(f"  • Repository: {self.repo_dir}\n")

        print(f"Next Steps:\n")
        print(f"1. Check training results:")
        print(f"   import json")
        print(f"   with open('{self.models_dir}/metadata_v7_opt.json') as f:")
        print(f"       results = json.load(f)")
        print(f"       for k, v in sorted(results.items()):")
        print(f"           print(f\"{{k}}: MAPE={{v['val_mape']:.2f}}% MAE={{v['val_mae']:.6f}}\")\n")
        
        print(f"2. Visualize predictions:")
        print(f"   !python {self.repo_dir}/visualize_predictions.py --list-klines\n")
        print(f"   !python {self.repo_dir}/visualize_predictions.py \\")
        print(f"       --model_path {self.models_dir}/BTCUSDT_15m_v7_opt.keras \\")
        print(f"       --symbol BTCUSDT_15m\n")

        print(f"3. Compare with baseline:")
        print(f"   # Check if metadata_v7.json exists (baseline version)")
        print(f"   import json")
        print(f"   with open('{self.models_dir}/metadata_v7_opt.json') as f:")
        print(f"       opt = json.load(f)")
        print(f"   try:")
        print(f"       with open('{self.models_dir}/metadata_v7.json') as f:")
        print(f"           baseline = json.load(f)")
        print(f"       for k in opt.keys():")
        print(f"           if k in baseline:")
        print(f"               opt_mape = opt[k]['val_mape']")
        print(f"               base_mape = baseline[k]['val_mape']")
        print(f"               improvement = (base_mape - opt_mape) / base_mape * 100")
        print(f"               print(f\"{{k}}: {{improvement:+.1f}}% improvement\")")
        print(f"   except: pass\n")

        print(f"4. Upload models to Hugging Face (optional):")
        print(f"   !python {self.repo_dir}/upload_models_template.py \\")
        print(f"       --models_dir {self.models_dir} \\")
        print(f"       --remote_folder models_v7_optimized\n")

    def run(self):
        """
        執行完整工作流程
        """
        self.print_section("Crypto V7 Optimized Models - Complete Colab Workflow", level=1)

        try:
            self.step_1_git_clone()
            self.step_2_install_dependencies()
            self.step_3_verify_setup()
            self.step_4_prepare_directories()
            training_success = self.step_5_start_training()
            self.step_6_summary()

            if training_success:
                print(f"\n{'='*70}")
                print(f"All steps completed successfully!")
                print(f"✓ Models trained with optimized architecture")
                print(f"✓ Expected 40% better accuracy than baseline")
                print(f"{'='*70}\n")
                return 0
            else:
                print(f"\n{'='*70}")
                print(f"Training failed. Please check the error messages above.")
                print(f"{'='*70}\n")
                return 1

        except Exception as e:
            print(f"\n✗ Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    workflow = CoLabWorkflow()
    exit_code = workflow.run()
    sys.exit(exit_code)
