import os
import json
from pathlib import Path
from huggingface_hub import HfApi

class HuggingFaceUploader:
    def __init__(self, hf_token=None, repo_id="zongowo111/cpb-models", models_dir='/content/all_models', repo_type="dataset"):
        """
        初始化 Hugging Face 上傳器
        
        Args:
            hf_token: Hugging Face API Token (如果未提供，將從環境變數讀取)
            repo_id: Hugging Face repo ID (format: username/repo-name)
            models_dir: 本地模型目錄路徑
            repo_type: repo 類型 ("model" 或 "dataset")
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found. Set it via environment variable or pass as argument.")
        
        self.repo_id = repo_id
        self.models_dir = models_dir
        self.repo_type = repo_type
        self.api = HfApi(token=self.hf_token)
    
    def get_keras_models(self):
        """
        獲取本地 all_models 目錄中所有的 .keras 模型檔案
        
        Returns:
            list: .keras 模型檔案路徑列表
        """
        models_path = Path(self.models_dir)
        if not models_path.exists():
            raise ValueError(f"Models directory not found: {self.models_dir}")
        
        keras_models = list(models_path.glob('*.keras'))
        print(f"\nFound {len(keras_models)} .keras models:")
        total_size = 0
        for model in sorted(keras_models):
            file_size = model.stat().st_size / (1024**2)  # 轉換為 MB
            total_size += file_size
            print(f"  - {model.name} ({file_size:.2f} MB)")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        return sorted(keras_models)
    
    def upload_folder(self, remote_folder="models_v7"):
        """
        一次性上傳整個 all_models 資料夾到 Hugging Face
        
        Args:
            remote_folder: Hugging Face repo 中的遠端資料夾名稱 (default: models_v7)
        """
        print(f"\n{'='*70}")
        print(f"Starting upload to Hugging Face")
        print(f"Repository: {self.repo_id}")
        print(f"Repository type: {self.repo_type}")
        print(f"Remote folder: {remote_folder}")
        print(f"{'='*70}\n")
        
        # 獲取所有 .keras 模型
        keras_models = self.get_keras_models()
        
        if not keras_models:
            print("\n✗ No .keras models found to upload")
            return False
        
        # 上傳整個資料夾
        print(f"\nUploading {len(keras_models)} models as a folder...\n")
        
        try:
            print("Uploading...", end=' ', flush=True)
            
            self.api.upload_folder(
                folder_path=self.models_dir,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                path_in_repo=remote_folder,
                ignore_patterns=["*.py", "*.md", "*.txt", "*.json", "*.csv"]  # 只上傳 .keras 檔案
            )
            
            print(f"✓")
            successful = True
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            successful = False
        
        # 輸出上傳結果總結
        print(f"\n{'='*70}")
        print(f"Upload Summary")
        print(f"{'='*70}")
        
        if successful:
            print(f"✓ Successfully uploaded {len(keras_models)} models")
            print(f"\n✓ All models available at:")
            print(f"  https://huggingface.co/datasets/{self.repo_id}/tree/main/{remote_folder}")
        else:
            print(f"✗ Upload failed")
        
        print(f"{'='*70}\n")
        
        return successful

if __name__ == '__main__':
    import sys
    
    # 設定參數
    hf_token = os.getenv('HF_TOKEN')  # 從環境變數讀取 token
    repo_id = "zongowo111/cpb-models"  # 上傳到 zongowo111/cpb-models dataset
    models_dir = '/content/all_models'
    remote_folder = 'models_v7'
    
    try:
        # 創建上傳器並執行上傳
        uploader = HuggingFaceUploader(
            hf_token=hf_token,
            repo_id=repo_id,
            models_dir=models_dir,
            repo_type="dataset"  # 上傳到 dataset repo
        )
        
        success = uploader.upload_folder(remote_folder=remote_folder)
        
        if success:
            print("\n✓ Upload process completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Upload process failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
