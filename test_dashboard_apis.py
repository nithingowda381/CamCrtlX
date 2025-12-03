#!/usr/bin/env python3

import requests
import json
import sys

def test_dashboard_apis():
    """Test all dashboard API endpoints"""
    base_url = 'http://localhost:5000'
    endpoints = [
        '/api/dashboard/metrics',
        '/api/dashboard/weekly', 
        '/api/dashboard/status',
        '/api/dashboard/activity'
    ]
    
    print("Testing Dashboard API Endpoints...")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            print(f"\nTesting {endpoint}:")
            response = requests.get(base_url + endpoint, timeout=5)
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Success: {data.get('success', False)}")
                if data.get('data'):
                    print(f"  Data keys: {list(data['data'].keys())}")
                    # Print some sample data
                    if isinstance(data['data'], dict):
                        for key, value in data['data'].items():
                            if key != 'lastUpdated':
                                print(f"    {key}: {value}")
            else:
                print(f"  Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"  Connection failed - Server may not be running")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Dashboard API Testing Complete")

if __name__ == "__main__":
    test_dashboard_apis()
