"""Run all tests (memory + tools)."""

import sys
import subprocess
import time


def check_server():
    """Check if server is running."""
    import requests
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    print("=" * 70)
    print("FASTAPI LANGGRAPH AGENT - COMPLETE TEST SUITE")
    print("=" * 70)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    if not check_server():
        print("❌ Server is not running!")
        print("\nPlease start the server:")
        print("  python -m uvicorn app.main:app --reload")
        print("\nOr using make:")
        print("  make dev")
        sys.exit(1)
    
    print("✅ Server is running")
    
    # Run memory tests
    print("\n" + "=" * 70)
    print("PART 1: MEMORY SYSTEM TESTS")
    print("=" * 70)
    result1 = subprocess.run([sys.executable, "tests/test_memory_system.py"])
    
    time.sleep(3)
    
    # Run tool tests
    print("\n" + "=" * 70)
    print("PART 2: TOOL FUNCTIONALITY TESTS")
    print("=" * 70)
    result2 = subprocess.run([sys.executable, "tests/test_tools.py"])
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    
    if result1.returncode == 0 and result2.returncode == 0:
        print("✅ All test suites completed successfully")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed - check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()

