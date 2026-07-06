import os
import sys
import time
import threading
import queue
import cv2
from PIL import Image
import mss
import pygetwindow as gw
import customtkinter as ctk
import numpy as np
import ctypes
import re

# Add workspace path to system path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vision import PokerVision
from core.table_state import TableState
from core.evaluator import PokerEvaluator
from core.decision import PokerDecisionEngine
from core.state_machine import PokerStateMachine
from core.action_executor import ActionExecutor, EmergencyAbortException
from core.xml_tracker import XMLTracker

# Win32 structures & helper functions for PID tracking
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
    ]

def get_visible_windows_with_pids():
    """Returns a list of tuples (pid, hwnd, title) of visible windows."""
    user32 = ctypes.windll.user32
    windows = []

    def callback(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value
                
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                windows.append((pid.value, hwnd, title))
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    # Filter out empty or common system windows to keep list clean
    filtered = []
    ignored_titles = {"Program Manager", "Settings", "Microsoft Text Input Application", "PHPHelp"}
    for pid, hwnd, title in windows:
        if title not in ignored_titles and not title.startswith("explorer"):
            filtered.append((pid, hwnd, title))
    return sorted(filtered, key=lambda x: x[2].lower())

def get_window_by_pid(pid):
    """Finds HWND of a visible window belonging to process id (PID)."""
    user32 = ctypes.windll.user32
    target_hwnd = [None]
    
    def callback(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            w_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(w_pid))
            if w_pid.value == pid:
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    target_hwnd[0] = hwnd
                    return False # stop enumerating
        return True
        
    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return target_hwnd[0]

def get_window_rect(hwnd):
    rect = RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

def activate_window(hwnd):
    user32 = ctypes.windll.user32
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9) # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")  # Modern premium styling

class PHPHelpApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("PHPHelp")
        self.geometry("1024x720")
        self.resizable(False, False)
        
        # Internal components
        self.vision = PokerVision()
        self.evaluator = PokerEvaluator()
        self.decision_engine = PokerDecisionEngine()
        self.action_executor = ActionExecutor()
        self.state_machine = PokerStateMachine(self)
        
        # Bot execution state
        self.bot_running = False
        self.bot_thread = None
        self.log_queue = queue.Queue()
        
        # Configuration variables
        self.mode_var = ctk.StringVar(value="Recommendation Only") # vs "Automatic Play"
        self.source_var = ctk.StringVar(value="Mock: 1_postflop_first_fold_check_raise.png") # vs board4.png, Live Game
        self.opponents_var = ctk.IntVar(value=1)
        self.simulations_var = ctk.IntVar(value=2000)
        self.target_window_var = ctk.StringVar(value="Bet365")
        
        # Toggleable decision layers
        self.layer_preflop_var = ctk.BooleanVar(value=True)
        self.layer_math_var = ctk.BooleanVar(value=True)
        self.layer_bluff_var = ctk.BooleanVar(value=True)
        self.layer_sizing_var = ctk.BooleanVar(value=True)
        
        # Turn Diagnostics variables
        self.last_raw_img = None
        self.last_table_state = None
        self.last_equity = None
        self.last_decision = None
        self.recent_logs = []
        self.last_valid_hero_stack = 760  # Tracks last valid stack to tolerate timer overlays
        self.table_state = TableState()
        self.xml_tracker = XMLTracker()
        self.pending_baseline_stacks = None
        
        # Set up UI Layout
        self.create_widgets()
        
        # Start queue poller and keyboard shortcut poller
        self.poll_log_queue()
        self.poll_keyboard_shortcuts()
        
    def create_widgets(self):
        # Grid Configuration (1 row, 2 columns: Sidebar + Main Content)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1) # Sidebar
        self.grid_columnconfigure(1, weight=3) # Main area
        
        # ==========================================
        # SIDEBAR (Control & Configuration)
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(11, weight=1) # spacer
        
        # App Title
        self.title_label = ctk.CTkLabel(self.sidebar, text="PHP HELP", font=ctk.CTkFont(size=22, weight="bold", family="Outfit"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sub_label = ctk.CTkLabel(self.sidebar, text="PHP Syntax Parser v1.0", font=ctk.CTkFont(size=12, slant="italic"))
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Bot Toggle Button
        self.start_btn = ctk.CTkButton(self.sidebar, text="START BOT", fg_color="#2eb85c", hover_color="#229647", font=ctk.CTkFont(weight="bold"), command=self.toggle_bot)
        self.start_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Mode Selection
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Execution Mode:", anchor="w")
        self.mode_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.mode_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["Recommendation Only", "Automatic Play"], variable=self.mode_var)
        self.mode_dropdown.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        
        # Source Selection
        self.source_label = ctk.CTkLabel(self.sidebar, text="Input Source:", anchor="w")
        self.source_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.source_dropdown = ctk.CTkOptionMenu(self.sidebar, values=[
            "Mock: 1_postflop_first_fold_check_raise.png", 
            "Mock: 2_postflop_river_fold_call_raise.png", 
            "Mock: 3_preflop_fold_allin.png", 
            "Mock: 4_postflop_river_fold_call_raise_facing_bet.png",
            "Mock: 5_flop_8sQh_check.png",
            "Mock: 6_flop_8sQh_check.png",
            "Mock: 7_preflop_5cKh_check.png",
            "Mock: 8_river_5cKh_check.png",
            "Mock: 9_preflop_5dJd_raise_pot_70.png",
            "Mock: 10_flop_5dJd_fold.png",
            "Mock: 11_flop_7c6s_check.png",
            "Mock: 12_preflop_Ad4h_fold.png",
            "Mock: 13_preflop_Ad4h_fold.png",
            "Mock: 14_preflop_Jh7h_fold.png",
            "Mock: 15_preflop_2sQc_fold.png",
            "Mock: 16_preflop_Kc6d_fold.png",
            "Live Capture"
        ], variable=self.source_var, command=self.on_source_changed)
        self.source_dropdown.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        # Target Window (Only visible when Live Capture selected)
        self.window_label = ctk.CTkLabel(self.sidebar, text="Target Window Name:", anchor="w")
        self.window_combo = ctk.CTkComboBox(self.sidebar, values=["Bet365"], variable=self.target_window_var)
        
        # Simulation Settings
        self.opponents_label = ctk.CTkLabel(self.sidebar, text="Opponents Range (1-5):", anchor="w")
        self.opponents_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.opponents_slider = ctk.CTkSlider(self.sidebar, from_=1, to=5, number_of_steps=4, variable=self.opponents_var, command=self.update_slider_labels)
        self.opponents_slider.grid(row=8, column=0, padx=20, pady=5, sticky="ew")
        self.opp_val_label = ctk.CTkLabel(self.sidebar, text="1 Active Opponent", font=ctk.CTkFont(size=11))
        self.opp_val_label.grid(row=9, column=0, padx=20, pady=(0, 10))
        
        # Compiler Modules (Decision Layers)
        self.layers_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.layers_frame.grid(row=10, column=0, padx=20, pady=(5, 5), sticky="ew")
        
        self.layers_title = ctk.CTkLabel(self.layers_frame, text="Active Modules:", font=ctk.CTkFont(weight="bold", size=12), anchor="w")
        self.layers_title.pack(anchor="w", pady=(0, 3))
        
        self.cb_preflop = ctk.CTkCheckBox(self.layers_frame, text="Preflop Range Engine", variable=self.layer_preflop_var, font=ctk.CTkFont(size=11))
        self.cb_preflop.pack(anchor="w", pady=1)
        
        self.cb_math = ctk.CTkCheckBox(self.layers_frame, text="Postflop EV Engine", variable=self.layer_math_var, font=ctk.CTkFont(size=11))
        self.cb_math.pack(anchor="w", pady=1)
        
        self.cb_bluff = ctk.CTkCheckBox(self.layers_frame, text="Bluffing Engine", variable=self.layer_bluff_var, font=ctk.CTkFont(size=11))
        self.cb_bluff.pack(anchor="w", pady=1)
        
        self.cb_sizing = ctk.CTkCheckBox(self.layers_frame, text="Pot-Sizing Shortcuts", variable=self.layer_sizing_var, font=ctk.CTkFont(size=11))
        self.cb_sizing.pack(anchor="w", pady=1)
        
        # Model Selection Dropdown packed inside layers_frame
        self.model_label = ctk.CTkLabel(self.layers_frame, text="Decision Model:", font=ctk.CTkFont(weight="bold", size=12), anchor="w")
        self.model_label.pack(anchor="w", pady=(15, 3))
        
        self.model_dropdown = ctk.CTkOptionMenu(
            self.layers_frame, 
            values=["Heuristic (Rules)", "XGBoost Classifier", "XGBoost Mixed (Pro + Human)", "PyTorch Neural Net"],
            command=self.on_model_changed
        )
        self.model_dropdown.pack(fill="x", pady=2)
        
        # Mixed Strategy Ratio Slider (packed dynamically only when Mixed is selected)
        self.ratio_label = ctk.CTkLabel(self.layers_frame, text="Mix: 50% Pro / 50% Human", font=ctk.CTkFont(size=11), anchor="w")
        self.ratio_slider = ctk.CTkSlider(
            self.layers_frame,
            from_=0.0,
            to=1.0,
            command=self.on_ratio_slider_changed
        )
        self.ratio_slider.set(0.5)
        
        # State Machine indicator at bottom of sidebar
        self.state_frame = ctk.CTkFrame(self.sidebar, height=45, fg_color="#1e2129", corner_radius=8)
        self.state_frame.grid(row=12, column=0, padx=20, pady=20, sticky="ew")
        self.state_frame.grid_propagate(False)
        self.state_frame.grid_columnconfigure(0, weight=1)
        self.state_frame.grid_rowconfigure(0, weight=1)
        self.state_lbl = ctk.CTkLabel(self.state_frame, text="STATUS: IDLE", font=ctk.CTkFont(size=13, weight="bold"))
        self.state_lbl.grid(row=0, column=0)
        
        # ==========================================
        # MAIN AREA (Dashboard & Logs)
        # ==========================================
        self.main_area = ctk.CTkFrame(self, fg_color="#0f1115", corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(0, weight=1) # visual board display
        self.main_area.grid_rowconfigure(1, weight=1) # decision & logs
        
        # 1. Top Section: Visual State Display Frame
        self.visual_frame = ctk.CTkFrame(self.main_area, fg_color="#181a20", corner_radius=12)
        self.visual_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.setup_visuals()
        
        # 2. Bottom Section: Logs Frame
        self.log_frame = ctk.CTkFrame(self.main_area, fg_color="#181a20", corner_radius=12)
        self.log_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.setup_logs()

    def setup_visuals(self):
        # Dividers inside visual frame
        self.visual_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.visual_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Title of telemetry
        self.telemetry_title = ctk.CTkLabel(self.visual_frame, text="Live Table Telemetry", font=ctk.CTkFont(size=15, weight="bold"))
        self.telemetry_title.grid(row=0, column=0, columnspan=3, pady=(10, 5))
        
        # Left Panel: Cards detected
        self.cards_panel = ctk.CTkFrame(self.visual_frame, fg_color="#1f222a", corner_radius=8)
        self.cards_panel.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")
        self.cards_panel.grid_columnconfigure(0, weight=1)
        
        self.card_lbl_title = ctk.CTkLabel(self.cards_panel, text="CARDS DETECTED", font=ctk.CTkFont(weight="bold", size=12))
        self.card_lbl_title.pack(pady=5)
        
        self.hero_cards_val = ctk.CTkLabel(self.cards_panel, text="Hero Hand: [--, --]", font=ctk.CTkFont(size=14, weight="bold"))
        self.hero_cards_val.pack(pady=5)
        
        self.comm_cards_val = ctk.CTkLabel(self.cards_panel, text="Board Cards: [--, --, --, --, --]", font=ctk.CTkFont(size=13))
        self.comm_cards_val.pack(pady=5)
        
        self.opps_stack_val = ctk.CTkLabel(self.cards_panel, text="Active Opponents: 0", font=ctk.CTkFont(size=12))
        self.opps_stack_val.pack(pady=5)
        
        # Center Panel: Seating Table Layout (Visual Grid)
        self.table_panel = ctk.CTkFrame(self.visual_frame, fg_color="#14161d", corner_radius=10)
        self.table_panel.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.table_panel.grid_rowconfigure((0, 1, 2), weight=1)
        self.table_panel.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.seat_widgets = {}
        
        # 3x3 Grid Seating Map
        seat_positions = {
            'seat_2': (0, 0, "Seat 2 (Top-Left)"),
            'seat_3': (0, 1, "Seat 3 (Top-Mid)"),
            'seat_4': (0, 2, "Seat 4 (Top-Right)"),
            'seat_1': (1, 0, "Seat 1 (Left)"),
            'pot': (1, 1, "POT"),
            'seat_5': (1, 2, "Seat 5 (Right)"),
            'hero': (2, 1, "Hero (Bottom)")
        }
        
        for key, (r, c, title) in seat_positions.items():
            if key == 'pot':
                # Center pot display
                pot_frame = ctk.CTkFrame(self.table_panel, fg_color="#1f222a", corner_radius=6, border_width=1, border_color="#ffd700")
                pot_frame.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                
                lbl_title = ctk.CTkLabel(pot_frame, text="POT", font=ctk.CTkFont(size=9, weight="bold"))
                lbl_title.pack(pady=(2, 0))
                
                self.pot_val = ctk.CTkLabel(pot_frame, text="0", text_color="#ffd700", font=ctk.CTkFont(size=13, weight="bold"))
                self.pot_val.pack(pady=(0, 2))
            else:
                # Player seat display frame
                frame = ctk.CTkFrame(self.table_panel, fg_color="#1f222a", corner_radius=6)
                frame.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                
                lbl_pos = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=8, weight="bold", slant="italic"), text_color="#8a90a0")
                lbl_pos.pack(pady=(1, 0))
                
                lbl_name = ctk.CTkLabel(frame, text="Empty", font=ctk.CTkFont(size=10, weight="bold"))
                lbl_name.pack(pady=0)
                
                lbl_stack = ctk.CTkLabel(frame, text="-", font=ctk.CTkFont(size=9))
                lbl_stack.pack(pady=(0, 1))
                
                vpip_agg_frame = ctk.CTkFrame(frame, fg_color="transparent")
                vpip_agg_frame.pack(pady=(0, 2))
                
                lbl_vpip = ctk.CTkLabel(vpip_agg_frame, text="●", font=ctk.CTkFont(size=8), text_color="#1f222a")
                lbl_vpip.pack(side="left", padx=2)
                
                lbl_agg = ctk.CTkLabel(vpip_agg_frame, text="●", font=ctk.CTkFont(size=8), text_color="#1f222a")
                lbl_agg.pack(side="left", padx=2)
                
                self.seat_widgets[key] = {
                    'frame': frame,
                    'pos': lbl_pos,
                    'name': lbl_name,
                    'stack': lbl_stack,
                    'vpip': lbl_vpip,
                    'agg': lbl_agg
                }
        
        # Right Panel: Win Probability / Equity
        self.equity_panel = ctk.CTkFrame(self.visual_frame, fg_color="#1f222a", corner_radius=8)
        self.equity_panel.grid(row=1, column=2, padx=15, pady=15, sticky="nsew")
        self.equity_panel.grid_columnconfigure(0, weight=1)
        
        self.equity_lbl_title = ctk.CTkLabel(self.equity_panel, text="WIN PROBABILITY", font=ctk.CTkFont(weight="bold", size=12))
        self.equity_lbl_title.pack(pady=(10, 2))
        
        self.equity_val = ctk.CTkLabel(self.equity_panel, text="0.0%", text_color="#3399ff", font=ctk.CTkFont(size=28, weight="bold"))
        self.equity_val.pack(pady=2)
        
        self.equity_desc = ctk.CTkLabel(self.equity_panel, text="W: 0.0%, D: 0.0%, L: 0.0%", font=ctk.CTkFont(size=11), text_color="#a0a0a0")
        self.equity_desc.pack(pady=2)

        # Recommended Action
        self.action_title_lbl = ctk.CTkLabel(self.equity_panel, text="RECOMMENDED ACTION", font=ctk.CTkFont(weight="bold", size=11), text_color="#8a90a0")
        self.action_title_lbl.pack(pady=(15, 2))

        self.action_val = ctk.CTkLabel(self.equity_panel, text="WAITING...", text_color="#ffd700", font=ctk.CTkFont(size=20, weight="bold"))
        self.action_val.pack(pady=5)
        
        self.action_reason_lbl = ctk.CTkLabel(self.equity_panel, text="-", font=ctk.CTkFont(size=10), text_color="#8a90a0", wraplength=180)
        self.action_reason_lbl.pack(pady=2)

    def setup_logs(self):
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(1, weight=1)
        
        # Log Title
        self.log_title = ctk.CTkLabel(self.log_frame, text="PHPHelp Output Streams", font=ctk.CTkFont(size=13, weight="bold"))
        self.log_title.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")
        
        # Flag Turn Button
        self.flag_btn = ctk.CTkButton(self.log_frame, text="Flag Turn (F12)", fg_color="#e55353", hover_color="#d93737", width=120, height=24, font=ctk.CTkFont(size=11, weight="bold"), command=self.save_diagnostics)
        self.flag_btn.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="e")
        
        # Log Text Box
        self.log_text = ctk.CTkTextbox(self.log_frame, wrap="word")
        self.log_text.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.log_text.configure(state="disabled") # read-only until we append

    def update_slider_labels(self, val):
        self.opp_val_label.configure(text=f"{int(val)} Active Opponents")

    def refresh_window_list(self):
        try:
            # Get all active window titles with PIDs
            wins = get_visible_windows_with_pids()
            titles = [f"[{pid}] {title}" for pid, hwnd, title in wins]
            
            curr = self.target_window_var.get()
            if curr and curr not in titles:
                # If current has no PID prefix, try to resolve it and update it
                match = re.match(r"^\[(\d+)\]", curr)
                if not match:
                    for pid, hwnd, title in wins:
                        if curr.lower() in title.lower():
                            curr = f"[{pid}] {title}"
                            self.target_window_var.set(curr)
                            break
                if curr not in titles:
                    titles.insert(0, curr)
            if not titles:
                titles = ["Bet365"]
            self.window_combo.configure(values=titles)
        except Exception as e:
            self.append_log(f"[SYSTEM] Error refreshing window list: {e}")

    def on_source_changed(self, val):
        if val == "Live Capture":
            self.refresh_window_list()
            # show target window field in sidebar
            self.window_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
            self.window_combo.grid(row=8, column=0, padx=20, pady=5, sticky="ew")
            # shift opponent slider down
            self.opponents_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
            self.opponents_slider.grid(row=10, column=0, padx=20, pady=5, sticky="ew")
            self.opp_val_label.grid(row=11, column=0, padx=20, pady=(0, 10))
            # shift layers and state down
            self.layers_frame.grid(row=12, column=0, padx=20, pady=(10, 0), sticky="ew")
            self.state_frame.grid(row=14, column=0, padx=20, pady=20, sticky="ew")
            self.sidebar.grid_rowconfigure(11, weight=0) # remove old spacer
            self.sidebar.grid_rowconfigure(13, weight=1) # set new spacer
        else:
            # hide target window field
            self.window_label.grid_forget()
            self.window_combo.grid_forget()
            # restore opponent slider row positions
            self.opponents_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
            self.opponents_slider.grid(row=8, column=0, padx=20, pady=5, sticky="ew")
            self.opp_val_label.grid(row=9, column=0, padx=20, pady=(0, 10))
            # restore layers and state positions
            self.layers_frame.grid(row=10, column=0, padx=20, pady=(10, 0), sticky="ew")
            self.state_frame.grid(row=12, column=0, padx=20, pady=20, sticky="ew")
            self.sidebar.grid_rowconfigure(13, weight=0) # remove old spacer
            self.sidebar.grid_rowconfigure(11, weight=1) # set new spacer

    def on_model_changed(self, choice):
        self.decision_engine.set_active_model(choice)
        self.append_log(f"[Decision] Switched active decision engine to: {choice}")
        if choice == "XGBoost Mixed (Pro + Human)":
            self.ratio_label.pack(anchor="w", pady=(10, 0))
            self.ratio_slider.pack(fill="x", pady=5)
        else:
            try:
                self.ratio_label.pack_forget()
                self.ratio_slider.pack_forget()
            except Exception:
                pass

    def on_ratio_slider_changed(self, val):
        self.decision_engine.mixed_ratio = float(val)
        pro_pct = int((1.0 - val) * 100)
        human_pct = int(val * 100)
        self.ratio_label.configure(text=f"Mix: {pro_pct}% Pro / {human_pct}% Human")

    def on_state_updated(self, state):
        """Callback from state machine to update the GUI status indicator."""
        # Use main thread execution since transitions might callbacks from background thread
        self.after(0, self._set_status_label, state)

    def _set_status_label(self, state):
        self.state_lbl.configure(text=f"STATUS: {state}")
        # Color state frames
        if state == "IDLE":
            self.state_frame.configure(fg_color="#1e2129")
        elif state == "WAITING_FOR_TURN":
            self.state_frame.configure(fg_color="#2b5c8f")
        elif state == "READING_STATE":
            self.state_frame.configure(fg_color="#d68f1c")
        elif state == "DECIDING":
            self.state_frame.configure(fg_color="#6f42c1")
        elif state == "EXECUTING_ACTION":
            self.state_frame.configure(fg_color="#28a745")

    def append_log(self, msg):
        self.log_queue.put(msg)
        # Store in rolling buffer for diagnostics
        self.recent_logs.append(msg)
        if len(self.recent_logs) > 100:
            self.recent_logs.pop(0)

    def poll_log_queue(self):
        """Polls the log queue and updates the GUI textbox."""
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.configure(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        # schedule next poll in 100ms
        self.after(100, self.poll_log_queue)

    def toggle_bot(self):
        if not self.bot_running:
            # Start the Bot
            self.bot_running = True
            self.start_btn.configure(text="STOP BOT", fg_color="#dc3545", hover_color="#bd2130")
            self.append_log("[SYSTEM] Starting PHPHelp engine...")
            
            # Start background worker thread
            self.state_machine.start()
            self.bot_thread = threading.Thread(target=self.bot_worker_loop, daemon=True)
            self.bot_thread.start()
        else:
            # Stop the Bot
            self.bot_running = False
            self.start_btn.configure(text="START BOT", fg_color="#2eb85c", hover_color="#229647")
            self.append_log("[SYSTEM] Stopping PHPHelp engine...")
            self.state_machine.stop()

    # ==========================================
    # BACKGROUND WORKER LOOP (Main logic)
    # ==========================================
    def bot_worker_loop(self):
        """Background thread executing the screenshot, CV, equity, decision, and click loop."""
        # Setup coordinates/rect for capturing
        mss_instance = mss.MSS()
        
        self.table_state.reset()
        
        while self.bot_running:
            try:
                # Load screenshot (Mock vs Live)
                source = self.source_var.get()
                img = None
                win_pos = (0, 0)
                win_size = None
                
                if source.startswith("Mock:"):
                    # Load mock image from disk
                    filename = source.replace("Mock: ", "").strip()
                    path = os.path.join("board_samples", filename)
                    if not os.path.exists(path):
                        path = filename # root fallback
                    img = cv2.imread(path)
                    if img is None:
                        self.append_log(f"[ERROR] Mock image {filename} not found!")
                        self.state_machine.error_occurred()
                        time.sleep(2)
                        continue
                else:
                    # Live Capture mode: find target window by PID or Title
                    target_input = self.target_window_var.get()
                    match = re.match(r"^\[(\d+)\]", target_input)
                    target_pid = None
                    hwnd = None
                    
                    if match:
                        target_pid = int(match.group(1))
                        hwnd = get_window_by_pid(target_pid)
                    else:
                        # Fallback: search all visible windows for a title match
                        all_wins = get_visible_windows_with_pids()
                        for p, h, t in all_wins:
                            if target_input.lower() in t.lower():
                                target_pid = p
                                hwnd = h
                                self.target_window_var.set(f"[{p}] {t}")
                                self.append_log(f"[SYSTEM] Resolved window '{target_input}' to PID {p}")
                                break
                                
                    if hwnd is None and target_pid:
                        # Re-scan by PID (title might have changed)
                        hwnd = get_window_by_pid(target_pid)
                        
                    if hwnd is None:
                        self.append_log(f"[WARNING] Target window '{target_input}' not found. Capture full screen.")
                        monitor = mss_instance.monitors[1]
                        sct_img = mss_instance.grab(monitor)
                        img = np.array(sct_img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    else:
                        # Restore window if minimized and bring to foreground
                        try:
                            activate_window(hwnd)
                        except Exception:
                            pass
                            
                        # Get window coordinates and crop
                        left, top, width, height = get_window_rect(hwnd)
                        win_pos = (left, top)
                        win_size = (width, height)
                        
                        monitor = {"top": top, "left": left, "width": width, "height": height}
                        sct_img = mss_instance.grab(monitor)
                        img = np.array(sct_img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
                # Verify image layout resolution is compatible (scale if needed)
                h, w = img.shape[:2]
                if abs(w - 1536) > 50 or abs(h - 1090) > 50:
                    # Scale to standard 1536x1090 to match coordinates
                    img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
                
                # Store latest screenshot copy for debug / diagnostics
                self.last_raw_img = img.copy() if img is not None else None
                
                # --- CONTINUOUS TRACKING ---
                # Read raw state from vision
                raw_state = self.vision.read_board_state(img)
                
                # Check for hand reset
                if self.table_state.detect_hand_reset(raw_state):
                    self.append_log("[SYSTEM] New hand detected. Resetting table state history.")
                    self.table_state.reset()
                    
                    if not source.startswith("Mock:"):
                        try:
                            baseline_stacks, hero_name = self.xml_tracker.get_baseline_stacks()
                            if baseline_stacks:
                                self.append_log(f"[XML Tracker] Loaded baseline stacks: {baseline_stacks} (Hero: {hero_name})")
                                self.pending_baseline_stacks = (baseline_stacks, hero_name)
                        except Exception as e:
                            self.append_log(f"[XML Tracker] Error loading baseline stacks: {e}")
                            self.pending_baseline_stacks = None
                    else:
                        self.pending_baseline_stacks = None
                    
                # Update stabilized table state datamodel
                self.table_state.update(raw_state)
                
                # Apply baseline stacks seeding on the first frame of a new hand (only for Live Capture)
                if not source.startswith("Mock:") and self.pending_baseline_stacks:
                    baseline_stacks, hero_name = self.pending_baseline_stacks
                    self.table_state.seed_stacks(baseline_stacks, hero_name)
                    self.pending_baseline_stacks = None
                    
                stabilized_state = self.table_state.to_dict()
                
                # Update GUI visual elements continuously
                self.after(0, self.update_telemetry_ui, stabilized_state)
                
                # Check if it's Hero's turn (look for active buttons)
                button_matches = self.vision.match_templates_in_roi(
                    img, self.vision.rois['buttons'], self.vision.button_templates, threshold=0.85, max_matches=1
                )
                
                if not button_matches:
                    # Not our turn. 
                    # If we were in a mid-state (e.g. DECIDING), we safely reset to WAITING_FOR_TURN
                    if self.state_machine.state not in ['IDLE', 'WAITING_FOR_TURN']:
                        self.state_machine.error_occurred()
                        
                    self.after(0, self.update_action_ui, "WAITING...", "Not Hero's turn", 0)
                    
                    # Sleep before next continuous tracking frame
                    time.sleep(1.5 if not source.startswith("Mock:") else 5.0)
                    continue
                    
                # It IS our turn!
                if self.state_machine.state == 'WAITING_FOR_TURN':
                    fold_btn_coord = button_matches[0][1] # relative coordinates
                    self.state_machine.turn_detected()
                    self.append_log("\n--- HERO TURN DETECTED ---")
                    
                    active_opps = [opp for opp in stabilized_state['opponents'].values() if opp.get('is_active', True)]
                    self.state_machine.state_read_complete()
                    
                    # 4. State Machine: DECIDING
                    # Check if we have cards recognized
                    if len(stabilized_state['hero_cards']) < 2:
                        self.last_table_state = stabilized_state
                        self.last_equity = 0.0
                        self.last_decision = ("PARSE_FAILED", "Local registers empty (Hero cards parse failed)", 0.0)
                        self.append_log("[Decision] Error: Local registers empty. Parse failed.")
                        self.state_machine.error_occurred()
                        time.sleep(2)
                        continue
                        
                    # Calculate Win Equity via Monte Carlo
                    self.append_log("[Decision] Parsing thread execution path...")
                    # Automatically use detected opponents count, fallback to GUI slider if none found
                    detected_opponents = len(active_opps)
                    if detected_opponents > 0:
                        num_opponents = detected_opponents
                        self.opponents_var.set(detected_opponents)
                        self.after(0, self.update_slider_labels, detected_opponents)
                    else:
                        num_opponents = self.opponents_var.get()
                    num_sims = self.simulations_var.get()
                    
                    equity, sim_msg = self.evaluator.calculate_equity(
                        stabilized_state['community_cards'],
                        stabilized_state['hero_cards'],
                        num_opponents=num_opponents,
                        num_simulations=num_sims
                    )
                    self.append_log(f"[Decision] {sim_msg}")
                    self.after(0, self.update_equity_ui, equity, sim_msg)
                    
                    # Determine is_preflop
                    is_preflop = len(stabilized_state['community_cards']) == 0
                    
                    # Track and fall back on valid hero stack size to tolerate timer overlays
                    hero_stack = stabilized_state['hero_stack']
                    if hero_stack > 0:
                        self.last_valid_hero_stack = hero_stack
                    else:
                        hero_stack = self.last_valid_hero_stack
                        self.append_log(f"[Vision] Stack size OCR obscured. Falling back to: {hero_stack}")

                    # Calculate decision parameters
                    pot = stabilized_state['pot_size']
                    
                    # Check if the second (Check/Call) and third (Bet/Raise) buttons are visible/available on screen
                    check_call_available = True
                    bet_raise_available = True
                    
                    # In mock mode, we manually set check_call_available to False for the fold-allin test case
                    if "3_preflop_fold_allin" in source:
                        check_call_available = False
                        
                    if not source.startswith("Mock:"):
                        fold_x, fold_y = fold_btn_coord
                        
                        # 1. Check Check/Call button presence
                        cc_crop = img[fold_y+20:fold_y+70, fold_x+210:fold_x+350]
                        if cc_crop.size > 0:
                            cc_val = np.mean(cc_crop, axis=(0,1))[1]
                            if cc_val < 60.0:
                                check_call_available = False
                                self.append_log("[Vision] Check/Call button is unavailable (all-in situation).")
                                
                        # 2. Check Bet/Raise button presence
                        btn_crop = img[fold_y+20:fold_y+70, fold_x+380:fold_x+520]
                        if btn_crop.size > 0:
                            green_val = np.mean(btn_crop, axis=(0,1))[1]
                            if green_val < 60.0:
                                bet_raise_available = False
                                self.append_log("[Vision] Bet/Raise button is unavailable (greyed out/hidden).")

                # Parse call amount from Check/Call button or mock files
                call_amount = 0.0
                cc_text = ""
                text_upper = ""
                
                if not source.startswith("Mock:"):
                    if check_call_available:
                        fold_x, fold_y = fold_btn_coord
                        # Crop and OCR Check/Call button
                        cc_text = self.vision.ocr_roi(img, (fold_x + 190, fold_y + 15, 160, 60))
                        text_upper = cc_text.upper().replace(',', '.')
                        
                        if any(w in text_upper for w in ["KALD", "CALL", "KLD", "KND"]):
                            match = re.search(r'(\d+(?:\.\d+)?)', text_upper)
                            if match:
                                try:
                                    call_amount = float(match.group(1))
                                except ValueError:
                                    call_amount = 2.0
                            else:
                                call_amount = 2.0
                            self.append_log(f"[Vision] Facing bet! Parsed Call Amount: {call_amount}")
                        else:
                            self.append_log(f"[Vision] No bet detected on Check/Call button. Text: '{cc_text}'")
                    else:
                        # If call button is unavailable (e.g. all-in situation), force facing a large bet
                        call_amount = 100.0
                else:
                    # Mock Mode fallback
                    if "board4" in source or "2_postflop_river" in source:
                        call_amount = 40.0
                    elif "3_preflop_fold_allin" in source:
                        call_amount = 100.0
                    elif "4_postflop_river" in source:
                        call_amount = 186.0

                action, reason, bet_size = self.decision_engine.make_decision(
                    stabilized_state['community_cards'],
                    stabilized_state['hero_cards'],
                    equity=equity,
                    pot_size=pot,
                    call_amount=call_amount,
                    hero_stack=hero_stack,
                    num_opponents=num_opponents,
                    is_preflop=is_preflop,
                    use_preflop_chart=self.layer_preflop_var.get(),
                    use_math_engine=self.layer_math_var.get(),
                    use_bluff_engine=self.layer_bluff_var.get(),
                    use_dynamic_sizing=self.layer_sizing_var.get(),
                    bet_raise_available=bet_raise_available,
                    check_call_available=check_call_available,
                    active_opponents=active_opps
                )
                
                # Save parsed states for debug / diagnostics
                self.last_table_state = stabilized_state
                self.last_equity = equity
                self.last_decision = (action, reason, bet_size)
                
                # Safeguard: If decided action is CHECK, but we detected the middle button as KALD/CALL or it's unavailable,
                # override to FOLD to prevent accidental calling.
                if action == 'CHECK' and not source.startswith("Mock:"):
                    if not check_call_available or any(w in text_upper for w in ["KALD", "CALL", "KLD", "KND"]):
                        self.append_log("[Safeguard] WARNING: Decided CHECK but Check/Call button is KALD or unavailable! Overriding to FOLD.")
                        action = 'FOLD'
                        reason = f"Safeguard: Blocked accidental call on CHECK decision. (OCR: '{cc_text}', Avail: {check_call_available})"
                        # Re-save decision with safeguard applied
                        self.last_decision = (action, reason, bet_size)
                
                self.append_log(f"[Decision] DECIDED BRANCH: **{action}**")
                self.append_log(f"[Decision] Reason: {reason}")
                if bet_size > 0:
                    self.append_log(f"[Decision] Size Allocation: {bet_size} units")
                    
                self.after(0, self.update_action_ui, action, reason, bet_size)
                self.state_machine.decision_made()
                
                # 5. State Machine: EXECUTING_ACTION
                execution_mode = self.mode_var.get()
                if execution_mode == "Automatic Play":
                    self.append_log(f"[Automation] Invoking native call for {action}...")
                    # Click button relative to fold button
                    success = self.action_executor.click_button_relative(
                        fold_btn_coord=fold_btn_coord,
                        action_type=action,
                        window_pos=win_pos,
                        window_size=win_size
                    )
                    if success:
                        self.append_log("[Automation] Thread interrupt completed successfully.")
                    else:
                        self.append_log("[Automation] Thread interrupt failed.")
                else:
                    self.append_log("[Automation] Logging only. Discarding mouse interrupt.")
                    time.sleep(1.5) # Simulate delay
                    
                # Cycle finished! Transition back to WAITING_FOR_TURN
                self.state_machine.action_completed()
                self.append_log("[SYSTEM] Parsing loop complete. Monitoring stream...")
                
                # Sleep between loops to avoid running immediately again
                time.sleep(5.0 if source.startswith("Mock:") else 1.0)
                
            except EmergencyAbortException:
                # Caught during mouse movements
                self.bot_running = False
                self.after(0, lambda: self.start_btn.configure(text="START BOT", fg_color="#2eb85c", hover_color="#229647"))
                self.append_log("[SYSTEM] Emergency Abort: Escape key pressed! Releasing mouse control.")
                self.state_machine.stop()
                time.sleep(1.0)
            except Exception as e:
                self.append_log(f"[CRITICAL ERROR] Execution Exception: {e}")
                import traceback
                self.append_log(traceback.format_exc())
                self.state_machine.error_occurred()
                time.sleep(3.0)

    def update_telemetry_ui(self, state):
        self.hero_cards_val.configure(text=f"Hero Hand: {state['hero_cards']}")
        self.comm_cards_val.configure(text=f"Community Cards: {state['community_cards']}")
        self.pot_val.configure(text=f"{state['pot_size']}")
        
        # 1. Update Hero Seat
        hero_widget = self.seat_widgets['hero']
        hero_stack = state['hero_stack']
        if hero_stack > 0:
            hero_widget['name'].configure(text="Hero")
            hero_widget['stack'].configure(text=f"{hero_stack} chips")
            hero_widget['frame'].configure(fg_color="#1b4d3e") # Active green background
            hero_widget['name'].configure(text_color="#2eb85c")
            hero_widget['stack'].configure(text_color="#2eb85c")
        else:
            hero_widget['name'].configure(text="Hero")
            hero_widget['stack'].configure(text="0")
            hero_widget['frame'].configure(fg_color="#2d3038") # Folded
            hero_widget['name'].configure(text_color="#8a90a0")
            hero_widget['stack'].configure(text_color="#8a90a0")
            
        # 2. Update Opponent Seats
        opponents = state.get('opponents', {})
        active_count = 0
        
        for i in range(1, 6):
            seat_key = f'seat_{i}'
            widget = self.seat_widgets[seat_key]
            
            if seat_key in opponents:
                opp = opponents[seat_key]
                name = opp['name']
                stack = opp['stack']
                state_lbl = opp.get('state', 'Active')
                is_active = opp.get('is_active', True)
                
                widget['name'].configure(text=name)
                
                if state_lbl == 'All-In':
                    widget['stack'].configure(text="ALL-IN", text_color="#ffd700")
                    widget['frame'].configure(fg_color="#4d3e1b") # gold/yellow
                    widget['name'].configure(text_color="#ffd700")
                    active_count += 1
                elif state_lbl == 'Folded':
                    widget['stack'].configure(text="Folded", text_color="#8a90a0")
                    widget['frame'].configure(fg_color="#2d3038") # gray
                    widget['name'].configure(text_color="#8a90a0")
                else: # Active
                    widget['stack'].configure(text=f"{stack} chips", text_color="#2eb85c")
                    widget['frame'].configure(fg_color="#1b4d3e") # green
                    widget['name'].configure(text_color="#2eb85c")
                    active_count += 1
                    
                # Update VPIP and AGG colors
                color_map = {
                    'Blue': "#3399ff",
                    'Green': "#2eb85c",
                    'Yellow': "#ffd700",
                    'Red': "#e55353",
                    None: "#1f222a" # Hidden
                }
                vpip_c = opp.get('vpip_color')
                agg_c = opp.get('agg_color')
                
                if 'vpip' in widget and 'agg' in widget:
                    widget['vpip'].configure(text_color=color_map.get(vpip_c, "#1f222a"))
                    widget['agg'].configure(text_color=color_map.get(agg_c, "#1f222a"))
            else:
                # Seat is empty / not detected
                widget['name'].configure(text="Empty", text_color="#4e5361")
                widget['stack'].configure(text="-", text_color="#4e5361")
                widget['frame'].configure(fg_color="#1f222a") # default dark frame
                if 'vpip' in widget and 'agg' in widget:
                    widget['vpip'].configure(text_color="#1f222a")
                    widget['agg'].configure(text_color="#1f222a")
                
        self.opps_stack_val.configure(text=f"Active Opponents: {active_count}")

    def update_equity_ui(self, equity, sim_msg=None):
        self.equity_val.configure(text=f"{equity * 100:.1f}%")
        # Color based on equity strength
        if equity > 0.65:
            self.equity_val.configure(text_color="#2eb85c") # Green (Very strong)
        elif equity > 0.45:
            self.equity_val.configure(text_color="#3399ff") # Blue (Medium)
        else:
            self.equity_val.configure(text_color="#e55353") # Red (Weak)
            
        if sim_msg:
            # We already format sim_msg correctly in evaluator.py
            # Expected format from evaluator: Simulated 2000 hands: W=45.0%, D=5.0%, L=50.0%
            # Just extract the part after the colon
            parts = sim_msg.split(":", 1)
            desc_text = parts[1].strip() if len(parts) > 1 else sim_msg
            self.equity_desc.configure(text=desc_text)

    def update_action_ui(self, action, reason, bet_size):
        # Format action text nicely
        text = action
        if bet_size > 0:
            text = f"{action} ({bet_size})"
            
        self.action_val.configure(text=text)
        
        # Color based on action type
        act_upper = action.upper()
        if "RAISE" in act_upper or "BET" in act_upper:
            self.action_val.configure(text_color="#ffd700") # Gold/Yellow
        elif "CALL" in act_upper or "CHECK" in act_upper:
            self.action_val.configure(text_color="#2eb85c") # Green
        elif "FOLD" in act_upper:
            self.action_val.configure(text_color="#e55353") # Red
        else:
            self.action_val.configure(text_color="#a0a0a0")
            
        # Clean up reason text for display
        clean_reason = reason
        if ":" in clean_reason:
            clean_reason = clean_reason.split(":")[-1].strip()
        self.action_reason_lbl.configure(text=clean_reason)

    def poll_keyboard_shortcuts(self):
        try:
            import ctypes
            
            # Virtual key code for ESC is 0x1B
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                if self.bot_running:
                    self.bot_running = False
                    self.after(0, lambda: self.start_btn.configure(text="START BOT", fg_color="#2eb85c", hover_color="#229647"))
                    self.append_log("[SYSTEM] Emergency Abort: Escape key pressed. Stopping bot.")
                    self.state_machine.stop()
                    
            # Virtual key code for F12 is 0x7B
            state_f12 = ctypes.windll.user32.GetAsyncKeyState(0x7B)
            # If the most significant bit is set, the key is currently down
            if state_f12 & 0x8000:
                if not hasattr(self, '_f12_was_down') or not self._f12_was_down:
                    self._f12_was_down = True
                    self.save_diagnostics()
            else:
                self._f12_was_down = False
        except Exception:
            pass
        self.after(100, self.poll_keyboard_shortcuts)

    def save_diagnostics(self):
        if self.last_raw_img is None:
            self.append_log("[SYSTEM] Warning: No active turn data to save.")
            return
            
        import datetime
        import json
        
        try:
            # Create timestamped directory in workspace root
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"diagnostics/turn_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Save screenshot
            screenshot_path = os.path.join(dir_name, "screenshot.png")
            cv2.imwrite(screenshot_path, self.last_raw_img)
            
            # Save telemetry metadata & decision output
            diag_data = {
                "timestamp": timestamp,
                "table_state": self.last_table_state,
                "equity": self.last_equity,
                "decision": self.last_decision
            }
            with open(os.path.join(dir_name, "telemetry.json"), "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=4, default=str)
                
            # Save recent log history
            with open(os.path.join(dir_name, "logs.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self.recent_logs))
                
            self.append_log(f"[SYSTEM] Turn flagged! Saved diagnostics to: {dir_name}/")
        except Exception as e:
            self.append_log(f"[ERROR] Failed to save diagnostics: {e}")

if __name__ == "__main__":
    app = PHPHelpApp()
    app.mainloop()
