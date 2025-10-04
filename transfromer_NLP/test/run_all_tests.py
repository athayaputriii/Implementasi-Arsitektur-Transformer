import sys
import os
import subprocess

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_test_simple(test_file):
    """Run test using subprocess"""
    try:
        print(f"Running {test_file}...")
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(__file__),  # Run from test directory
            encoding='utf-8',  # Force UTF-8 encoding
            errors='replace'   # Replace problematic characters
        )
        
        if result.returncode == 0:
            print(f"PASS: {test_file}")
            # Print output without special characters
            if result.stdout.strip():
                # Clean output by removing/replacing special chars
                clean_output = result.stdout.encode('ascii', 'replace').decode('ascii')
                lines = clean_output.strip().split('\n')
                for line in lines[-3:]:  # Print last 3 lines
                    print(f"   {line}")
            return True
        else:
            print(f"FAIL: {test_file} (return code: {result.returncode})")
            if result.stdout.strip():
                clean_stdout = result.stdout.encode('ascii', 'replace').decode('ascii')
                print("STDOUT:", clean_stdout[-500:])  # Last 500 chars
            if result.stderr.strip():
                clean_stderr = result.stderr.encode('ascii', 'replace').decode('ascii')
                print("STDERR:", clean_stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test_file}")
        return False
    except Exception as e:
        print(f"CRASH: {test_file} - {e}")
        return False

def main():
    print("Running All Transformer Tests")
    print("=" * 40)
    
    test_files = [
        "test_embedding.py",
        "test_attention.py", 
        "test_feed_forward.py",
        "test_layer_norm.py",
        "test_decoder.py",
        "test_integration.py"
    ]
    
    # Filter existing files
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("ERROR: No test files found in test/ directory!")
        return 1
    
    print(f"Found {len(existing_files)} test files")
    print()
    
    passed = 0
    for test_file in existing_files:
        if run_test_simple(test_file):
            passed += 1
        print()
    
    print("=" * 40)
    print(f"RESULTS: {passed}/{len(existing_files)} passed")
    
    if passed == len(existing_files):
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("FAILURE: Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())