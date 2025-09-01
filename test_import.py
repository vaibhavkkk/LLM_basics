# Test script to debug the issue
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)

try:
    import requests
    print("✅ requests imported successfully")
    print("requests location:", requests.__file__)
except ImportError as e:
    print("❌ Import error:", e)