import subprocess

def get_gpu_memory_usage():
    try:
        # Use nvidia-smi to get the memory info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print("Error: Unable to run nvidia-smi. Make sure NVIDIA drivers are installed.")
            print(result.stderr)
            return
        
        # Parse the output
        memory_info = result.stdout.strip().split("\n")
        for i, line in enumerate(memory_info):
            used, total = map(int, line.split(","))
            available = total - used
            print(f"GPU {i}:")
            print(f"  Used Memory: {used} MB")
            print(f"  Total Memory: {total} MB")
            print(f"  Available Memory: {available} MB")
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Ensure you have an NVIDIA GPU and drivers installed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    get_gpu_memory_usage()
