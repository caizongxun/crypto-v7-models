#!/usr/bin/env python3
"""
完整 Colab 工作流程

Description:
    此腳本包含從 git clone 到訓練完成的完整工作流程。
    適合在 Google Colab 中直接執行。

Usage in Colab:
    # 在 Colab cell 中執行
    !curl -s https://raw.githubusercontent.com/caizongxun/crypto-v7-models/main/colab_complete_workflow.py | python

    或者直接複製以下內容到 Colab cell 中執行：
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

        # 檢查是否已經 clone
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

        # 升級 pip
        self.run_command("pip install --upgrade pip", "Upgrade pip")

        # 安裝必要的套件
        packages = [
            "tensorflow>=2.13.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "yfinance>=0.2.0",
            "huggingface_hub>=0.16.0",
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
            import yfinance
            print(f"✓ yfinance installed")
        except ImportError:
            print("✗ yfinance not found")

        try:
            from huggingface_hub import HfApi
            print(f"✓ huggingface_hub installed")
        except ImportError:
            print("✗ huggingface_hub not found")

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
        第 5 步: 開始訓練
        """
        self.print_section("Step 5: Start Model Training", level=1)

        print(f"Training configuration:")
        print(f"  • Models directory: {self.models_dir}")
        print(f"  • Klines directory: {self.klines_dir}")
        print(f"  • Total cryptos: 20")
        print(f"  • Total timeframes: 2 (15m, 1h)")
        print(f"  • Total models to train: 40")
        print()
        print(f"Starting training...\n")

        # 執行訓練
        train_script = f"""
import sys
sys.path.insert(0, '{self.repo_dir}')

exec(open('{self.repo_dir}/train_v7_main.py').read())

pipeline = TrainingPipeline(output_dir='{self.models_dir}', klines_dir='{self.klines_dir}')
pipeline.train_all_models()
"""

        try:
            exec(train_script)
            print(f"\n✓ Training completed successfully!")
            return True
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_6_summary(self):
        """
        第 6 步: 總結
        """
        self.print_section("Step 6: Summary", level=1)

        print(f"✓ Workflow completed!\n")

        print(f"Generated files:")
        print(f"  • Models: {self.models_dir}")
        print(f"  • Klines: {self.klines_dir}")
        print(f"  • Repository: {self.repo_dir}\n")

        print(f"Next steps:\n")
        print(f"1. Visualize predictions:")
        print(
            f"   !python {self.repo_dir}/visualize_predictions.py --list-klines\n"
        )
        print(f"   !python {self.repo_dir}/visualize_predictions.py \\")
        print(f"       --model_path {self.models_dir}/BTCUSDT_15m_v7.keras \\")
        print(f"       --symbol BTCUSDT_15m\n")

        print(f"2. Upload models to Hugging Face:")
        print(f"   !python {self.repo_dir}/upload_models_template.py \\")
        print(f"       --models_dir {self.models_dir} \\")
        print(f"       --remote_folder models_v7\n")

        print(f"3. Check metadata:")
        print(f"   import json")
        print(f"   with open('{self.models_dir}/metadata_v7.json') as f:")
        print(f"       metadata = json.load(f)")
        print(f"       for key, value in metadata.items():")
        print(f"           print(f'{{key}}: {{value[\'val_mape\']:.2f}}%')\n")

    def run(self):
        """
        執行完整工作流程
        """
        self.print_section("Crypto V7 Models - Complete Colab Workflow", level=1)

        try:
            # 步驟 1: Git Clone
            self.step_1_git_clone()

            # 步驟 2: 安裝依賴
            self.step_2_install_dependencies()

            # 步驟 3: 驗證環境
            self.step_3_verify_setup()

            # 步驟 4: 準備目錄
            self.step_4_prepare_directories()

            # 步驟 5: 開始訓練
            training_success = self.step_5_start_training()

            # 步驟 6: 總結
            self.step_6_summary()

            if training_success:
                print(f"\n{'='*70}")
                print(f"All steps completed successfully!")
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
