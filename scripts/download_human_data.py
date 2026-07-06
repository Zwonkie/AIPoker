import os
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

def download_file(url, dest_path):
    try:
        # URL encode spaces
        encoded_url = url.replace(" ", "%20")
        req = urllib.request.Request(encoded_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(content)
        return True
    except Exception as e:
        # print(f"Error downloading {url}: {e}")
        return False

def main():
    print("[Ingestion] Starting Human Poker Ingestion...")
    raw_dir = os.path.join("data", "raw_human_hands")
    os.makedirs(raw_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/uoftcprg/phh-dataset/main/data/handhq/PS-2009-07-01_2009-07-23_25NLH_OBFU/0.25"
    
    download_tasks = []
    # Download 30 files of human play session data
    for file_id in range(1, 31):
        filename = f"ps NLH handhq_{file_id}-OBFUSCATED.phhs"
        url = f"{base_url}/{filename}"
        dest_path = os.path.join(raw_dir, filename)
        download_tasks.append((url, dest_path))
        
    print(f"[Ingestion] Queued {len(download_tasks)} human files for download...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, task[0], task[1]) for task in download_tasks]
        for f in futures:
            if f.result():
                success_count += 1
                
    print(f"[Ingestion] Complete! Successfully downloaded {success_count} human hand history files to data/raw_human_hands/")

if __name__ == '__main__':
    main()
