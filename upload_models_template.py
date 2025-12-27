#!/usr/bin/env python3
"""
加密貨幣模紋上傳端

Description:
    此昇文件是一個可重複使用的上傳窗樣，
    可下提供給 v8、v9 等未來版本使用。

Usage:
    python upload_models_template.py \
        --hf_token YOUR_HF_TOKEN \
        --repo_id zongowo111/cpb-models \
        --models_dir /path/to/models \
        --remote_folder models_v8 \
        --repo_type dataset

Environment Variables:
    HF_TOKEN: Hugging Face API Token (if not provided via --hf_token)

Example for v8:
    python upload_models_template.py \
        --models_dir /content/all_models_v8 \
        --remote_folder models_v8
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


class ModelUploader:
    """
    一個通用的模紋上傳器類別，支持不同版本的模紋上傳
    """

    def __init__(
        self,
        hf_token: str,
        repo_id: str = "zongowo111/cpb-models",
        models_dir: str = "/content/all_models",
        repo_type: str = "dataset",
    ):
        """
        初始化模紋上傳器

        Args:
            hf_token (str): Hugging Face API Token
            repo_id (str): Hugging Face repo ID (username/repo-name)
            models_dir (str): 本地模紋目錄路徑
            repo_type (str): Repo 類型 ("model" 或 "dataset")
        """
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. Provide via --hf_token or set HF_TOKEN environment variable."
            )

        self.hf_token = hf_token
        self.repo_id = repo_id
        self.models_dir = models_dir
        self.repo_type = repo_type
        self.api = HfApi(token=self.hf_token)

        print(f"\n{'='*70}")
        print(f"Model Uploader Initialized")
        print(f"{'='*70}")
        print(f"Repository: {self.repo_id}")
        print(f"Repository type: {self.repo_type}")
        print(f"Local models directory: {self.models_dir}")
        print(f"{'='*70}\n")

    def get_keras_models(self) -> List[Path]:
        """
        獲取本地目錄中所有的 .keras 模紋檔案

        Returns:
            List[Path]: .keras 模紋檔案路徑列表

        Raises:
            ValueError: 如果目錄不存在
        """
        models_path = Path(self.models_dir)
        if not models_path.exists():
            raise ValueError(f"Models directory not found: {self.models_dir}")

        keras_models = list(models_path.glob("*.keras"))
        if not keras_models:
            raise ValueError(f"No .keras models found in {self.models_dir}")

        print(f"Found {len(keras_models)} .keras models:")
        total_size = 0
        for model in sorted(keras_models):
            file_size = model.stat().st_size / (1024**2)  # MB
            total_size += file_size
            print(f"  - {model.name} ({file_size:.2f} MB)")

        print(f"\nTotal size: {total_size:.2f} MB")
        return sorted(keras_models)

    def verify_metadata(self) -> Optional[dict]:
        """
        驗證是否存在 metadata 檔案

        Returns:
            Optional[dict]: Metadata 幣倒，或 None 如果不存在
        """
        metadata_path = Path(self.models_dir) / "metadata_v7.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"✓ Found metadata file with {len(metadata)} entries")
                return metadata
            except json.JSONDecodeError:
                print(f"✗ Failed to parse metadata file")
                return None
        else:
            print(f"ℹ No metadata file found (optional)")
            return None

    def upload_folder(self, remote_folder: str = "models_v7") -> bool:
        """
        一次性上傳整個模紋目錄到 Hugging Face

        Args:
            remote_folder (str): Hugging Face repo 中的遠端EE檔社名稱

        Returns:
            bool: 上傳成功上設失敗
        """
        print(f"\n{'='*70}")
        print(f"Starting upload to Hugging Face")
        print(f"Remote folder: {remote_folder}")
        print(f"{'='*70}\n")

        try:
            # 驗證檔案
            keras_models = self.get_keras_models()
            self.verify_metadata()

            print(f"\nUploading {len(keras_models)} models as a folder...")
            print("This may take a few minutes...\n")
            print("Uploading...", end=" ", flush=True)

            # 上傳整個目錄
            self.api.upload_folder(
                folder_path=self.models_dir,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                path_in_repo=remote_folder,
                ignore_patterns=[
                    "*.py",
                    "*.md",
                    "*.txt",
                    "*.json",
                    "*.csv",
                    ".DS_Store",
                    "__pycache__",
                ],  # 只上傳 .keras 檔案
            )

            print(f"✓")
            return True

        except Exception as e:
            print(f"✗")
            print(f"\nError during upload: {str(e)}")
            return False

    def display_summary(self, success: bool, remote_folder: str):
        """
        顯示上傳結果總結

        Args:
            success (bool): 上傳是否成功
            remote_folder (str): 遠端EE檔社名稱
        """
        print(f"\n{'='*70}")
        print(f"Upload Summary")
        print(f"{'='*70}")

        if success:
            print(f"✓ Successfully uploaded models!")
            print(f"\n✓ Models available at:")
            if self.repo_type == "dataset":
                url = f"https://huggingface.co/datasets/{self.repo_id}/tree/main/{remote_folder}"
            else:
                url = f"https://huggingface.co/{self.repo_id}/tree/main/{remote_folder}"
            print(f"  {url}")
        else:
            print(f"✗ Upload failed")

        print(f"{'='*70}\n")


def main():
    """
    主叨函數
    """
    parser = argparse.ArgumentParser(
        description="Upload .keras models to Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload v7 models
  python upload_models_template.py --models_dir /content/all_models --remote_folder models_v7

  # Upload v8 models
  python upload_models_template.py --models_dir /content/all_models_v8 --remote_folder models_v8

  # Custom repo
  python upload_models_template.py --repo_id username/custom-repo --models_dir /path/to/models --remote_folder my_models
        """,
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API Token (default: read from HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="zongowo111/cpb-models",
        help="Hugging Face repo ID (default: zongowo111/cpb-models)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/content/all_models",
        help="Local models directory (default: /content/all_models)",
    )
    parser.add_argument(
        "--remote_folder",
        type=str,
        default="models_v7",
        help="Remote folder name (default: models_v7)",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        choices=["model", "dataset"],
        default="dataset",
        help="Repository type (default: dataset)",
    )

    args = parser.parse_args()

    # 從環境變數讀取 token （如果未提供）
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    try:
        # 創建上傳器
        uploader = ModelUploader(
            hf_token=hf_token,
            repo_id=args.repo_id,
            models_dir=args.models_dir,
            repo_type=args.repo_type,
        )

        # 執行上傳
        success = uploader.upload_folder(remote_folder=args.remote_folder)

        # 顯示結果
        uploader.display_summary(success, args.remote_folder)

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
