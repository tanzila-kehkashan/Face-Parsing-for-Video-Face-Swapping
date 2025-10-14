#!/usr/bin/env python3
"""
FAKESYNCSTUDIO API Test Script
Test the API functionality with sample files
"""

import requests
import time
import json
import os
from pathlib import Path
import argparse

API_BASE_URL = "http://localhost:9876"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   API Version: {data.get('version', 'unknown')}")
            print(f"   Models Status: {data.get('models_status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_info():
    """Test the root endpoint"""
    print("\n📋 Testing API info...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Info: {data['name']} v{data['version']}")
            print(f"   Description: {data['description']}")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def test_file_upload(source_path, target_path, custom_settings=None):
    """Test file upload and processing"""
    print(f"\n📤 Testing file upload...")
    print(f"   Source: {source_path}")
    print(f"   Target: {target_path}")
    
    if not Path(source_path).exists():
        print(f"❌ Source file not found: {source_path}")
        return None
    
    if not Path(target_path).exists():
        print(f"❌ Target file not found: {target_path}")
        return None
    
    try:
        files = {
            'source_image': open(source_path, 'rb'),
            'target_video': open(target_path, 'rb')
        }
        
        data = {}
        if custom_settings:
            data['settings'] = json.dumps(custom_settings)
        
        print("   Uploading files...")
        response = requests.post(f"{API_BASE_URL}/upload/", files=files, data=data, timeout=30)
        
        # Close files
        files['source_image'].close()
        files['target_video'].close()
        
        if response.status_code == 202:
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"✅ Upload successful! Job ID: {job_id}")
            print(f"   Status: {job_data['status']}")
            print(f"   Features: {', '.join(job_data.get('features', []))}")
            return job_id
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_job_status(job_id):
    """Test job status checking"""
    print(f"\n📊 Testing job status for {job_id}...")
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/status/", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Status check successful: {status_data['status']}")
            if 'message' in status_data:
                print(f"   Message: {status_data['message']}")
            if 'file_size' in status_data:
                print(f"   File size: {status_data['file_size']:,} bytes")
            return status_data
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return None

def test_job_logs(job_id, duration=10):
    """Test job log streaming"""
    print(f"\n📜 Testing log streaming for {job_id} ({duration}s)...")
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/stream_logs", stream=True, timeout=30)
        if response.status_code == 200:
            print("✅ Log streaming started:")
            
            start_time = time.time()
            for line in response.iter_lines(decode_unicode=True):
                if time.time() - start_time > duration:
                    break
                    
                if line.startswith('data: '):
                    log_message = line[6:]  # Remove 'data: ' prefix
                    if log_message.strip():
                        print(f"   📝 {log_message}")
                elif line.startswith('event: end'):
                    print("   🏁 Processing completed")
                    break
            
            return True
        else:
            print(f"❌ Log streaming failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Log streaming error: {e}")
        return False

def wait_for_completion(job_id, max_wait=600, check_interval=10):
    """Wait for job completion"""
    print(f"\n⏳ Waiting for job completion (max {max_wait}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status_data = test_job_status(job_id)
        if not status_data:
            print("❌ Could not check status")
            return False
        
        status = status_data['status']
        if status == 'completed':
            print("🎉 Job completed successfully!")
            return True
        elif status == 'failed':
            print("❌ Job failed!")
            return False
        elif status in ['processing', 'uploading']:
            print(f"⏳ Still {status}... waiting {check_interval}s")
            time.sleep(check_interval)
        else:
            print(f"❓ Unknown status: {status}")
            time.sleep(check_interval)
    
    print(f"⏰ Timeout after {max_wait}s")
    return False

def test_result_download(job_id, output_path="test_result.mp4"):
    """Test result download"""
    print(f"\n⬇️ Testing result download for {job_id}...")
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/result", timeout=60)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = Path(output_path).stat().st_size
            print(f"✅ Download successful: {output_path} ({file_size:,} bytes)")
            return True
        else:
            print(f"❌ Download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def test_job_cleanup(job_id):
    """Test job deletion"""
    print(f"\n🗑️ Testing job cleanup for {job_id}...")
    try:
        response = requests.delete(f"{API_BASE_URL}/jobs/{job_id}", timeout=10)
        if response.status_code == 200:
            print("✅ Job deleted successfully")
            return True
        else:
            print(f"❌ Job deletion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Job deletion error: {e}")
        return False

def run_full_test(source_path, target_path, custom_settings=None, download_result=True, cleanup=True):
    """Run complete API test suite"""
    print("🎭 FAKESYNCSTUDIO API Full Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    if not test_health_check():
        print("❌ Health check failed - API may not be running")
        return False
    
    if not test_api_info():
        print("❌ API info failed")
        return False
    
    # Test file upload
    job_id = test_file_upload(source_path, target_path, custom_settings)
    if not job_id:
        print("❌ Upload test failed")
        return False
    
    # Test log streaming (brief)
    test_job_logs(job_id, duration=5)
    
    # Wait for completion
    if not wait_for_completion(job_id, max_wait=300):  # 5 minutes
        print("❌ Job did not complete in time")
        return False
    
    # Test result download
    if download_result:
        if not test_result_download(job_id, f"test_result_{job_id}.mp4"):
            print("❌ Download test failed")
            return False
    
    # Test cleanup
    if cleanup:
        test_job_cleanup(job_id)
    
    print("\n🎉 All tests completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test FAKESYNCSTUDIO API")
    parser.add_argument("--source", "-s", required=True, help="Source image path")
    parser.add_argument("--target", "-t", required=True, help="Target video path") 
    parser.add_argument("--api-url", default="http://localhost:9876", help="API base URL")
    parser.add_argument("--no-download", action="store_true", help="Skip result download")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip job cleanup")
    parser.add_argument("--settings", help="Custom settings JSON")
    parser.add_argument("--quick", action="store_true", help="Quick test (health + upload only)")
    
    args = parser.parse_args()
    
    global API_BASE_URL
    API_BASE_URL = args.api_url.rstrip('/')
    
    custom_settings = None
    if args.settings:
        try:
            custom_settings = json.loads(args.settings)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid settings JSON: {e}")
            return False
    
    if args.quick:
        # Quick test
        print("🚀 Quick API Test")
        print("=" * 30)
        success = (test_health_check() and 
                  test_api_info() and 
                  test_file_upload(args.source, args.target, custom_settings) is not None)
        print(f"\n{'✅ Quick test passed!' if success else '❌ Quick test failed!'}")
        return success
    else:
        # Full test
        return run_full_test(
            args.source, 
            args.target, 
            custom_settings,
            download_result=not args.no_download,
            cleanup=not args.no_cleanup
        )

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        exit(1)