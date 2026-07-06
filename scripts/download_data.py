import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

def download_file(url, dest_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(content)
        return True
    except Exception:
        return False

def main():
    print("[Ingestion] Starting Poker Ingestion...")
    raw_dir = os.path.join("data", "raw_hands")
    os.makedirs(raw_dir, exist_ok=True)
    
    # We will download up to 1000 hand files from Pluribus directories 90-99
    base_url = "https://raw.githubusercontent.com/uoftcprg/phh-dataset/main/data/pluribus"
    
    download_tasks = []
    # We iterate over some directories and file IDs
    for directory in range(80, 100):
        for file_id in range(150): # 150 files per directory = 3000 total files
            url = f"{base_url}/{directory}/{file_id}.phh"
            dest_name = f"pluribus_{directory}_{file_id}.phh"
            dest_path = os.path.join(raw_dir, dest_name)
            download_tasks.append((url, dest_path))
            
    print(f"[Ingestion] Queued {len(download_tasks)} files for download. Executing in parallel...")
    
    success_count = 0
    # Use ThreadPoolExecutor for fast downloading (parallelism)
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(download_file, task[0], task[1]) for task in download_tasks]
        for f in futures:
            if f.result():
                success_count += 1
                if success_count % 100 == 0:
                    print(f"[Ingestion] Downloaded {success_count} files...")
                    
    print(f"[Ingestion] Complete! Successfully downloaded {success_count} hand history files to data/raw_hands/")

if __name__ == '__main__':
    main()
