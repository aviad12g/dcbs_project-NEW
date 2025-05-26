#!/usr/bin/env python3
"""
Test script to check if Python is working correctly.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--test", help="Test argument")
    args = parser.parse_args()
    
    print("Python is working correctly!")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {args}")

if __name__ == "__main__":
    main() 