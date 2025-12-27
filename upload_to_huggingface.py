import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

class HuggingFaceUploader:
    def __init__(self, hf_token=None, repo_id="caizongxun/crypto-v7-models", models_dir='/content/all_models'):
        """
        初始化 Hugging Face 上傳器
        
        Args:
            hf_token: Hugging Face API Token (如果未提供，將從環境變數讀取)
            repo_id: Hugging Face repo ID (format: username/repo-name)
            models_dir: 本地模型目錄路徑
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found. Set it via environment variable or pass as argument.")
        
        self.repo_id = repo_id
        self.models_dir = models_dir
        self.api = HfApi(token=self.hf_token)
    
    def ensure_repo_exists(self):
        """
        確認 Hugging Face repo 存在，如果不存在則創建
        """
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            print(f"✓ Repo {self.repo_id} already exists")
        except Exception as e:
            print(f"✗ Repo not found: {e}")
            print(f"Creating repo {self.repo_id}...")
            try:
                create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    token=self.hf_token,
                    private=False
                )
                print(f"✓ Repo {self.repo_id} created successfully")
            except Exception as create_error:
                print(f"✗ Failed to create repo: {create_error}")
                raise
    
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
        for model in sorted(keras_models):
            print(f"  - {model.name}")
        
        return sorted(keras_models)
    
    def upload_models(self, remote_folder="models_v7"):
        """
        上傳所有 .keras 模型到 Hugging Face 的指定資料夾
        
        Args:
            remote_folder: Hugging Face repo 中的遠端資料夾名稱 (default: models_v7)
        """
        print(f"\n{'='*70}")
        print(f"Starting upload to Hugging Face")
        print(f"Repository: {self.repo_id}")
        print(f"Remote folder: {remote_folder}")
        print(f"{'='*70}\n")
        
        # 確認 repo 存在
        self.ensure_repo_exists()
        
        # 獲取所有 .keras 模型
        keras_models = self.get_keras_models()
        
        if not keras_models:
            print("\n✗ No .keras models found to upload")
            return False
        
        # 上傳每個模型
        successful_uploads = 0
        failed_uploads = []
        
        print(f"\nUploading {len(keras_models)} models...\n")
        
        for idx, model_path in enumerate(keras_models, 1):
            model_name = model_path.name
            remote_path = f"{remote_folder}/{model_name}"
            
            try:
                print(f"[{idx}/{len(keras_models)}] Uploading {model_name}...", end=' ', flush=True)
                
                self.api.upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=remote_path,
                    repo_id=self.repo_id,
                    repo_type="model"
                )
                
                print(f"✓")
                successful_uploads += 1
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:60]}")
                failed_uploads.append((model_name, str(e)))
        
        # 上傳 metadata 檔案（如果存在）
        metadata_path = Path(self.models_dir) / 'metadata_v7.json'
        if metadata_path.exists():
            try:
                print(f"\nUploading metadata_v7.json...", end=' ', flush=True)
                self.api.upload_file(
                    path_or_fileobj=str(metadata_path),
                    path_in_repo=f"{remote_folder}/metadata_v7.json",
                    repo_id=self.repo_id,
                    repo_type="model"
                )
                print(f"✓")
            except Exception as e:
                print(f"✗ Error: {str(e)[:60]}")
        
        # 輸出上傳結果總結
        print(f"\n{'='*70}")
        print(f"Upload Summary")
        print(f"{'='*70}")
        print(f"✓ Successfully uploaded: {successful_uploads}/{len(keras_models)} models")
        
        if failed_uploads:
            print(f"\n✗ Failed uploads ({len(failed_uploads)}):")
            for model_name, error in failed_uploads:
                print(f"  - {model_name}: {error[:50]}...")
        
        print(f"\n✓ All models available at:")
        print(f"  https://huggingface.co/{self.repo_id}/tree/main/{remote_folder}")
        print(f"{'='*70}\n")
        
        return successful_uploads > 0

if __name__ == '__main__':
    import sys
    
    # 設定參數
    hf_token = os.getenv('HF_TOKEN')  # 從環境變數讀取 token
    repo_id = "caizongxun/crypto-v7-models"  # 改成你的 repo ID
    models_dir = '/content/all_models'
    remote_folder = 'models_v7'
    
    try:
        # 創建上傳器並執行上傳
        uploader = HuggingFaceUploader(
            hf_token=hf_token,
            repo_id=repo_id,
            models_dir=models_dir
        )
        
        success = uploader.upload_models(remote_folder=remote_folder)
        
        if success:
            print("\n✓ Upload process completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Upload process failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
