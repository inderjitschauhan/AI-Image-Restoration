import os

def list_pth_files_table(root_dir):
    data = []
    total_size = 0

    # Scan all subfolders
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pth"):
                full_path = os.path.join(root, file)
                size_bytes = os.path.getsize(full_path)
                size_mb = size_bytes / (1024 * 1024)

                total_size += size_bytes

                data.append({
                    "name": file,
                    "path": full_path,
                    "size_mb": size_mb
                })

    # Sort by size (largest first)
    data.sort(key=lambda x: x["size_mb"], reverse=True)

    # Print table header
    print("\n" + "="*100)
    print(f"{'File Name':30} | {'Size (MB)':10} | {'Path'}")
    print("="*100)

    # Print rows
    for item in data:
        print(f"{item['name'][:30]:30} | {item['size_mb']:10.2f} | {item['path']}")

    print("="*100)
    print(f"{'TOTAL':30} | {total_size / (1024*1024):10.2f} MB")
    print("="*100)


# 👉 CHANGE THIS PATH if needed
WEIGHTS_DIR = r"weights"

list_pth_files_table(WEIGHTS_DIR)