#!/usr/bin/env python3
"""
Test script to diagnose Phi-3.5 model access issues
"""

from huggingface_hub import HfApi
import os

def test_phi_access():
    print("=== Testing Phi-3.5 Model Access ===")
    
    # Check token
    token = os.environ.get('HF_HUB_TOKEN')
    print(f"Token present: {bool(token)}")
    print(f"Token length: {len(token) if token else 0}")
    
    # Test API access
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('name', 'Unknown')}")
        
        # Check model info
        info = api.model_info('microsoft/Phi-3.5-mini-instruct')
        print(f"Model ID: {info.modelId}")
        print(f"Model gated: {getattr(info, 'gated', 'Unknown')}")
        print(f"Model private: {getattr(info, 'private', 'Unknown')}")
        
        # Test file access
        print("\nTesting direct file access...")
        from huggingface_hub import hf_hub_download
        
        config_path = hf_hub_download(
            repo_id="microsoft/Phi-3.5-mini-instruct",
            filename="config.json",
            cache_dir="./temp_cache"
        )
        print(f"Config downloaded to: {config_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_phi_access()
    if success:
        print("\n✓ All tests passed - model should be accessible")
    else:
        print("\n✗ Access test failed") 