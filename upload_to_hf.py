#!/usr/bin/env python3
"""
HuggingFace 上傳脚本
支援統一上傳所有模式（包括兩個 batch 的結果）
"""

from huggingface_hub import HfApi, HfFolder, login
from pathlib import Path
import os
import sys
import getpass

def upload_models_to_hf():
    print("="*80)
    print("HuggingFace 模式上傳工具")
    print("="*80)
    
    # 第一步：稽清 token
    print("\n【第一步】 驗證 HuggingFace Token")
    print("-" * 80)
    
    # 先檢查是否已經有有效 token
    try:
        saved_token = HfFolder.get_token()
        if saved_token:
            print(f"\u2713 偵測到已保存的 token")
            use_saved = input("\u662f否使用既有 token？ (y/n, 預設 y): ").strip().lower()
            if use_saved != 'n':
                token = saved_token
            else:
                token = getpass.getpass("\u8acb輸入新的 HuggingFace Token (https://huggingface.co/settings/tokens): ")
        else:
            token = getpass.getpass("\u8acb輸入 HuggingFace Token (https://huggingface.co/settings/tokens): ")
    except Exception as e:
        token = getpass.getpass("\u8acb輸入 HuggingFace Token (https://huggingface.co/settings/tokens): ")
    
    if not token:
        print("\u2717 Token 不能空白")
        return False
    
    # 驗證 token
    try:
        login(token=token, write_permission=True)
        print("\u2713 Token 驗證成功")
    except Exception as e:
        print(f"\u2717 Token 驗證失敗: {str(e)}")
        return False
    
    # 第二步：確認上傳路徑
    print("\n【第二步】 確認上傳設定")
    print("-" * 80)
    
    local_models_dir = "/content/all_models"
    repo_id = "zongowo111/cpb-models"
    remote_folder = "models_v8"
    
    print(f"\u672c地路徑: {local_models_dir}")
    print(f"Repository ID: {repo_id}")
    print(f"遠端路徑: {remote_folder}")
    
    # 確認本地路徑存在
    if not os.path.exists(local_models_dir):
        print(f"\u2717 本地路徑不存在: {local_models_dir}")
        return False
    
    # 統計要上傳的檔案
    file_count = len(list(Path(local_models_dir).rglob("*")))
    print(f"\u2713 找到 {file_count} 個檔案")
    
    # 第三步：上傳
    print("\n【第三步】 開始上傳")
    print("-" * 80)
    
    try:
        api = HfApi()
        
        print(f"\u6b63在上傳 {local_models_dir} 到 {repo_id}/{remote_folder}...\n")
        
        # 基本參數上傳
        repo_url = api.upload_folder(
            folder_path=local_models_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=remote_folder,
            commit_message="新增：V8 訓練完成的 40 個 LSTM 模型 (20 幣種 × 2 時間框架)"
        )
        
        print("\n" + "="*80)
        print("✓ 上傳完成！")
        print("="*80)
        print(f"\u2713 遮端位置: https://huggingface.co/datasets/{repo_id}/tree/main/{remote_folder}")
        print(f"\u2713 Commit URL: {repo_url}")
        print(f"\u2713 總自稚檔案数: {file_count}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\u2717 上傳失敗: {str(e)}")
        print(f"\u2717 錯誤類型: {type(e).__name__}")
        
        # 提供更詳的錯誤信息
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n是否是 Token 錯誤？請檢查：")
            print("1. Token 是否有 Write 權限")
            print("2. Token 是否未過期")
            print("3. Repository 是否存在")
            print("\n重新獲取 token: https://huggingface.co/settings/tokens")
        
        return False

if __name__ == "__main__":
    success = upload_models_to_hf()
    sys.exit(0 if success else 1)
