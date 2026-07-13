import sys
import os
import json
import re

def parse_training_log(logfile):
    if not os.path.exists(logfile):
        print(f"Error: Log file {logfile} not found.")
        return

    try:
        with open(logfile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(logfile, 'r', encoding='utf-16') as f:
            lines = f.readlines()

    # Find the last dashboard block
    start_idx = -1
    end_idx = -1

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith("+===============") and "Loss Q" in lines[i-1] if i > 0 else False:
            end_idx = i
        elif line.startswith("+===============") and "SELF-PLAY" in lines[i+1] if i < len(lines)-1 else False:
            if end_idx != -1:
                start_idx = i
                break

    if start_idx != -1 and end_idx != -1:
        dashboard = "".join(lines[start_idx:end_idx+1])
        print("```text\n" + dashboard.strip() + "\n```")
        
        # Parse into JSON
        telemetry = {
            "progress": "",
            "progress_percent": "",
            "epoch": "",
            "stage": "",
            "eta": "",
            "train_loss": "",
            "val_loss": "",
            "entropy": "",
            "seats": [],
            "equity_matrix": []
        }
        
        dash_lines = dashboard.strip().split("\n")
        for line in dash_lines:
            if "Active Personality:" in line:
                m = re.search(r"Active Personality:\s+(\w+)", line)
                if m: telemetry["personality"] = m.group(1).strip()
            elif "Hands Simulated:" in line:
                m = re.search(r"Hands Simulated:\s+([0-9,]+).*?\(\s*([\d.]+)%\)", line)
                if m:
                    telemetry["progress"] = m.group(1)
                    telemetry["progress_percent"] = m.group(2)
            elif "Training Epoch:" in line:
                m = re.search(r"Training Epoch:\s+(\d+)", line)
                if m: telemetry["epoch"] = m.group(1)
            elif "Curriculum Stage:" in line:
                m = re.search(r"Curriculum Stage:\s+(.*?)\s*\|", line)
                if m: telemetry["stage"] = m.group(1).strip()
            elif "ETA:" in line:
                m = re.search(r"ETA:\s+(.*?)\s*\|", line)
                if m: telemetry["eta"] = m.group(1).strip()
            elif "Action Entropy:" in line:
                m = re.search(r"Action Entropy:\s+([\d.]+)", line)
                if m: telemetry["entropy"] = m.group(1)
            elif "Train Loss" in line and "Val Loss" in line:
                m = re.search(r"Train Loss.*?:\s+([\d.]+).*?Val Loss:\s+([\d.]+)", line)
                if m:
                    telemetry["train_loss"] = m.group(1)
                    telemetry["val_loss"] = m.group(2)
            elif "Seat " in line and "BB/100" in line:
                m = re.search(r"-\s+(Seat\s+\d+\s+[^:]+):\s+([\+\-]?[\d.]+)\s+BB/100\s+\(VPIP:\s*([\d.]+)%\s+AGG:\s*([\d.]+)%\)\s+\[R:(\d+)\s+F:(\d+)\s+AI:(\d+)", line)
                if m:
                    telemetry["seats"].append({
                        "name": m.group(1).strip(),
                        "bb100": float(m.group(2)),
                        "vpip": float(m.group(3)),
                        "agg": float(m.group(4)),
                        "r": int(m.group(5)),
                        "f": int(m.group(6)),
                        "ai": int(m.group(7))
                    })
            elif "|  EXPLOITATION SCOREBOARD (Net BB/100 Matrix):" in line:
                if "exploitation_matrix" not in telemetry:
                    telemetry["exploitation_matrix"] = []
            elif "exploitation_matrix" in telemetry and line.startswith("|") and ("Hero " in line or "Seat " in line or "Opp " in line) and "Winner" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 7:
                    row_name = parts[0]
                    # Handle cell values (some might be '-')
                    def parse_val(v):
                        if v == '-': return 0.0
                        try: return float(v)
                        except: return 0.0
                    
                    telemetry["exploitation_matrix"].append({
                        "name": row_name,
                        "hero": parse_val(parts[1]),
                        "s1": parse_val(parts[2]),
                        "s2": parse_val(parts[3]),
                        "s3": parse_val(parts[4]),
                        "s4": parse_val(parts[5]),
                        "s5": parse_val(parts[6])
                    })
            elif "%" in line and ("<20%" in line or "20-40%" in line or "40-60%" in line or "60-80%" in line or ">80%" in line):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 10:
                    telemetry["equity_matrix"].append({
                        "bracket": parts[0],
                        "fold": parts[1],
                        "call": parts[2],
                        "raise": parts[3],
                        "rr": parts[4],
                        "all_in": parts[5],
                        "avg_end_street": parts[6],
                        "net_chips": parts[7],
                        "won_chips": parts[8],
                        "lost_chips": parts[9]
                    })
                elif len(parts) >= 8:
                    telemetry["equity_matrix"].append({
                        "bracket": parts[0],
                        "fold": parts[1],
                        "call": parts[2],
                        "raise": parts[3],
                        "rr": parts[4],
                        "all_in": parts[5],
                        "avg_end_street": parts[6],
                        "net_chips": parts[7]
                    })
                    
        # Write JSON to same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "telemetry.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(telemetry, jf, indent=2)
            
    else:
        print("Could not find a complete dashboard block in the log.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_training_log.py <logfile>")
    else:
        parse_training_log(sys.argv[1])
