import os

root_dir = r"c:\REPO\Antigravity\AIPoker"

dir_sizes = {}
file_sizes = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    parts = dirpath.split(os.sep)
    if '.git' in parts or '.venv' in parts or '__pycache__' in parts:
        continue
    dir_size = 0
    for f in filenames:
        fp = os.path.join(dirpath, f)
        try:
            size = os.path.getsize(fp)
            dir_size += size
            file_sizes.append((fp, size))
        except:
            pass
    dir_sizes[dirpath] = dir_size
    
# Propagate sizes up the tree
for d in sorted(dir_sizes.keys(), key=lambda x: x.count(os.sep), reverse=True):
    parent = os.path.dirname(d)
    if parent in dir_sizes:
        dir_sizes[parent] += dir_sizes[d]

print("Top Largest Folders (MB):")
for d, s in sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)[:15]:
    if s > 1024 * 1024: # > 1MB
        print(f"{d.replace(root_dir, '') or 'Root'}: {s / (1024*1024):.2f} MB")

print("\nTop Largest Files (MB):")
for f, s in sorted(file_sizes, key=lambda x: x[1], reverse=True)[:15]:
    if s > 1024 * 1024:
        print(f"{f.replace(root_dir, '')}: {s / (1024*1024):.2f} MB")
