import sys
import urllib3
import requests

print("--- Python Environment Check ---")
print(f"Python Executable: {sys.executable}")
print(f"requests version: {requests.__version__}")
print(f"urllib3 version: {urllib3.__version__}")
print(f"urllib3 file location: {urllib3.__file__}")
print("-----------------------------")