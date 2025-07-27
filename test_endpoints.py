#!/usr/bin/env python3
"""
Simple test script to verify the backend endpoints are working
"""

import requests
import json

# Base URL for the backend
BASE_URL = "http://18.130.24.251:8000"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health data: {json.dumps(data, indent=2)}")
        else:
            print(f"Health check failed: {response.text}")
    except Exception as e:
        print(f"Health check error: {e}")

def test_available_accents():
    """Test the available accents endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/available_accents")
        print(f"Available accents status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available accents: {data.get('accents', [])}")
        else:
            print(f"Available accents failed: {response.text}")
    except Exception as e:
        print(f"Available accents error: {e}")

if __name__ == "__main__":
    print("Testing backend endpoints...")
    print("=" * 50)
    
    test_health()
    print("-" * 30)
    test_available_accents()
    
    print("=" * 50)
    print("Test completed!") 