import glob
import os

templates_dir = r'c:\REPO\Antigravity\AIPoker\digit_templates_16x16'
print("Templates in directory:")
for path in glob.glob(os.path.join(templates_dir, '*.png')):
    print(f"  {os.path.basename(path)}")
