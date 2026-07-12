import os
import glob

history_dir = r"C:\Users\zwonk\AppData\Local\Poker at Bet365.DK\data\Zwonkie\History\Data\Tournaments"
if os.path.exists(history_dir):
    xml_files = glob.glob(os.path.join(history_dir, "*.xml"))
    if xml_files:
        xml_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest = xml_files[0]
        print(f"Latest XML file: {latest}")
        print("--- CONTENT ---")
        with open(latest, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            # Print the last 150 lines
            for line in lines[-150:]:
                print(line, end="")
    else:
        print("No XML files found")
else:
    print(f"Directory not found: {history_dir}")
