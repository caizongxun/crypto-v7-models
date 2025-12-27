#!/usr/bin/env python3
"""
將本地 klines/ 資料夾上傳到 Hugging Face Datasets

使用方式：
    huggingface-cli login  # 先登入
    python upload_to_hf.py

注意：
- 會創建/更新 HF dataset 中的 klines/ 資料夾
- 使用批量上傳避免觸發 API 限制
- 支援斷點續傳（已存在的檔案會跳過）
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("缺少 huggingface_hub，請先安裝：")
    print("  pip install huggingface_hub")
    sys.exit(1)

# 設定
LOCAL_KLINES_DIR = "klines"
HF_REPO_ID = "zongowo111/cpb-models"  # 改成你的 HF dataset ID
HF_REPO_TYPE = "dataset"
REMOTE_KLINES_PATH = "klines_binance_us"  # HF 上的遠端路徑
BATCH_SIZE = 50  # 每批上傳的檔案數


class HFKlinesUploader:
    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()
        self._verify_repo_access()

    def _verify_repo_access(self):
        """驗證是否有 repo 存取權"""
        try:
            repo_info = self.api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type)
            print(f"✓ 成功連接到 {self.repo_id}")
            print(f"  最後修改時間: {repo_info.last_modified}")
        except RepositoryNotFoundError:
            print(f"✗ 找不到 repo: {self.repo_id}")
            print("  請確認 HF_REPO_ID 設定正確，且已執行 'huggingface-cli login'")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 連接失敗: {str(e)}")
            sys.exit(1)

    def collect_files(self, local_dir: str) -> List[Tuple[str, str]]:
        """
        收集所有本地檔案
        
        Returns:
            List[(local_path, remote_path)]
        """
        files = []
        local_path = Path(local_dir)
        
        if not local_path.exists():
            print(f"✗ 本地目錄不存在: {local_dir}")
            return files
        
        # 遞迴收集所有檔案
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # 計算相對路徑
                rel_path = file_path.relative_to(local_path)
                remote_path = f"{REMOTE_KLINES_PATH}/{rel_path}"
                files.append((str(file_path), remote_path))
        
        return sorted(files)

    def upload_batch(self, file_pairs: List[Tuple[str, str]], batch_num: int, total_batches: int) -> bool:
        """
        上傳一批檔案
        
        Args:
            file_pairs: [(local_path, remote_path)] 列表
            batch_num: 目前批次編號
            total_batches: 總批次數
            
        Returns:
            True 如果成功，False 如果失敗
        """
        operations = []
        
        for local_path, remote_path in file_pairs:
            try:
                # 建立上傳操作
                operation = CommitOperationAdd(
                    path_in_repo=remote_path,
                    path_or_fileobj=local_path
                )
                operations.append(operation)
            except Exception as e:
                print(f"  ✗ 無法讀取 {local_path}: {str(e)[:80]}")
                return False
        
        if not operations:
            print(f"  ✗ 批次 {batch_num} 無有效檔案")
            return False
        
        try:
            # 執行批量上傳
            commit_info = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=operations,
                commit_message=f"Upload klines batch {batch_num}/{total_batches} (Binance US {len(operations)} files)"
            )
            print(f"  ✓ 批次 {batch_num}: 成功上傳 {len(operations)} 個檔案")
            print(f"    Commit: {commit_info.commit_url}")
            return True
        except Exception as e:
            print(f"  ✗ 批次 {batch_num} 上傳失敗: {str(e)[:100]}")
            return False

    def upload(self, local_dir: str = LOCAL_KLINES_DIR):
        """
        主上傳流程
        """
        print("="*70)
        print("=          Klines 批量上傳至 Hugging Face Dataset            =")
        print("="*70 + "\n")
        
        # 1. 收集檔案
        print(f"正在掃描本地檔案: {local_dir}/")
        files = self.collect_files(local_dir)
        
        if not files:
            print(f"✗ 找不到任何檔案")
            return
        
        print(f"✓ 找到 {len(files)} 個檔案")
        
        # 2. 計算批次
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"✓ 將分 {total_batches} 批上傳 (每批 {BATCH_SIZE} 個檔案)\n")
        
        # 3. 批量上傳
        successful_batches = 0
        failed_batches = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(files))
            batch_files = files[start_idx:end_idx]
            
            print(f"批次 {batch_idx + 1}/{total_batches}: 上傳檔案 {start_idx + 1}-{end_idx}")
            
            if self.upload_batch(batch_files, batch_idx + 1, total_batches):
                successful_batches += 1
            else:
                failed_batches.append(batch_idx + 1)
        
        # 4. 總結
        print("\n" + "="*70)
        print("=                         上傳完成                             =")
        print("="*70)
        print(f"成功上傳: {successful_batches}/{total_batches} 批")
        print(f"上傳檔案: {len(files)} 個")
        print(f"遠端路徑: {self.repo_id}/{REMOTE_KLINES_PATH}/")
        
        if failed_batches:
            print(f"\n⚠ 失敗批次: {failed_batches}")
            print(f"請檢查網路連線或檔案權限，然後重新執行")
        else:
            print(f"\n✓ 所有檔案已成功上傳到 Hugging Face Dataset")
            print(f"  查看: https://huggingface.co/datasets/{self.repo_id}/tree/main/{REMOTE_KLINES_PATH}")


def main():
    # 檢查本地檔案
    if not os.path.exists(LOCAL_KLINES_DIR):
        print(f"✗ 本地 {LOCAL_KLINES_DIR}/ 目錄不存在")
        print("  請先執行 download_klines_binance_us.py 下載資料")
        sys.exit(1)
    
    if not os.listdir(LOCAL_KLINES_DIR):
        print(f"✗ {LOCAL_KLINES_DIR}/ 目錄為空")
        sys.exit(1)
    
    # 開始上傳
    uploader = HFKlinesUploader(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    uploader.upload(local_dir=LOCAL_KLINES_DIR)


if __name__ == "__main__":
    main()
