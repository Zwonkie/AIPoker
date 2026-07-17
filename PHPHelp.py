import os
import sys
import time
import threading
import queue
import json
import datetime
import cv2
from PIL import Image
import mss
import pygetwindow as gw
import tkinter as tk
import customtkinter as ctk
import numpy as np
import ctypes
import re

# Add workspace path to system path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Canonical action ordering for diagnostics. Covers the v13 3-way {FOLD,CALL,RAISE} and the
# v14 6-way {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN} policies. F12 shows whichever keys exist.
ACTION_DIAG_ORDER = ("FOLD", "CALL", "RAISE", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")

# Short display labels for the live Action Distribution panel.
ACTION_DISPLAY_NAMES = {
    "FOLD": "Fold", "CALL": "Call", "RAISE": "Raise",
    "RAISE_33": "Raise 33%", "RAISE_66": "Raise 66%", "RAISE_POT": "Raise Pot", "ALLIN": "All-In",
}

# Fixed on-screen seat layout, in clockwise order (matches the 3x3 grid in setup_visuals: hero
# bottom -> seat_1 left -> seat_2 top-left -> seat_3 top-mid -> seat_4 top-right -> seat_5 right ->
# back to hero). Used only to APPROXIMATE action order (who's already acted this street vs who
# hasn't) for the equity panel's opponent-color breakdown -- not used by the equity math itself.
SEAT_ORDER_CLOCKWISE = ['hero', 'seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']

# Action Distribution bar colors: RAW = the actor's untouched softmax output; SAMPLED = the same
# distribution after live temperature-sharpening (core/decision.py), which is what the sampler
# ("dice roll") actually draws from. The two can diverge -- sharpening pulls weight away from
# already-unlikely actions toward the favorite. Where their bars overlap we blend the colors.
_DIST_RAW_RGB = (51, 153, 255)      # blue,  matches #3399ff
_DIST_SAMPLED_RGB = (255, 196, 0)   # gold,  matches #ffd700 (slightly deeper for contrast vs blend)


def _rgb_hex(rgb):
    return "#%02x%02x%02x" % rgb


def _blend_hex(rgb_a, rgb_b):
    return _rgb_hex(tuple((a + b) // 2 for a, b in zip(rgb_a, rgb_b)))

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
        self.geometry("1240x820")
        self.resizable(False, False)
        
        # Internal components
        self.vision = PokerVision()
        self.evaluator = PokerEvaluator()
        self.decision_engine = PokerDecisionEngine(game_type="nlh")
        self.action_executor = ActionExecutor()
        self.state_machine = PokerStateMachine(self)
        
        # Bot execution state
        self.bot_running = False
        self.bot_thread = None
        self.log_queue = queue.Queue()
        
        # Configuration variables
        self.mode_var = ctk.StringVar(value="Recommendation Only") # vs "Automatic Play"
        self.source_var = ctk.StringVar(value="Mock: 1_postflop_first_fold_check_raise.png") # vs board4.png, Live Game
        self.board_size_var = ctk.StringVar(value="6-Max")
        self.looseness_var = ctk.DoubleVar(value=0.0)
        self.opponents_var = ctk.IntVar(value=1)
        self.simulations_var = ctk.IntVar(value=2000)
        self.target_window_var = ctk.StringVar(value="Bet365")
        self.big_blind_var = ctk.DoubleVar(value=25.0)
        
        # Toggleable decision layers (default to False to see true model actions)
        self.layer_preflop_var = ctk.BooleanVar(value=False)
        self.layer_math_var = ctk.BooleanVar(value=False)
        self.layer_bluff_var = ctk.BooleanVar(value=False)
        self.layer_sizing_var = ctk.BooleanVar(value=False)
        
        # Turn Diagnostics variables
        self.last_raw_img = None
        self.last_table_state = None
        self.last_equity = None
        self.last_decision = None
        self.last_ev_dict = None       # full model output (policy probs + Q-values + decision path)
        self.last_equity_meta = None   # how equity was computed (range-aware vs random, opp colors)
        self.last_window_title = None  # source window title -> board id for the history session
        # Hold the last real decision on screen for a few seconds (or until the next real decision,
        # whichever comes first) instead of snapping back to "WAITING..." the moment it's no longer
        # Hero's turn -- gives enough time to actually read the equity/action/distribution panel.
        self.last_decision_ts = 0.0
        self.MIN_DECISION_DISPLAY_SECONDS = 10.0

        # Debounce: after acting, the real client can take a moment to visually clear the
        # fold/check/raise buttons (network/animation lag). Our re-check cadence is only ~1s, so
        # without this the very next frame can still see the SAME buttons and misread it as a
        # fresh "Hero's turn," re-deciding and re-acting on stale state -- a double-action bug.
        # Require at least one confirmed "not Hero's turn" frame after acting before acting again.
        self._awaiting_turn_clear = False

        # --- Continuous turn history (live-bot recorder) ------------------------------------
        # While the bot runs, EVERY decided turn is appended to history/<board_id>/turns.jsonl
        # (replay-ready). F12 flags a specific turn: it also saves the heavy screenshot + layered
        # summary under history/<board_id>/flagged/ and marks it in flags.jsonl. board_id is
        # derived from the window title (stakes stripped so blind changes don't fork the folder).
        self.session_board_id = None
        self.session_history_dir = None
        self.session_turn_count = 0
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
        # Grid Configuration (1 row, 2 columns: Sidebar + Main Content). Sidebar gets weight=0 --
        # it's sized to its own content's natural minimum (the longest dropdown entry) and none of
        # the window's surplus width; Main area (weight=1, the only nonzero column) absorbs
        # everything else, so shrinking the sidebar's content directly grows the dashboard.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0) # Sidebar
        self.grid_columnconfigure(1, weight=1) # Main area
        
        # ==========================================
        # SIDEBAR (Control & Configuration)
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(16, weight=1) # spacer
        
        # App Title
        self.title_label = ctk.CTkLabel(self.sidebar, text="PHP HELP", font=ctk.CTkFont(size=22, weight="bold", family="Outfit"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sub_label = ctk.CTkLabel(self.sidebar, text="PHP Syntax Parser v1.0", font=ctk.CTkFont(size=12, slant="italic"))
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Bot Toggle Button & Auto-Live Shortcut Frame
        self.btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.btn_frame.grid_columnconfigure(0, weight=3) # Start Bot (larger)
        self.btn_frame.grid_columnconfigure(1, weight=2) # Auto-Live (smaller)
        
        self.start_btn = ctk.CTkButton(self.btn_frame, text="START BOT (F5)", fg_color="#2eb85c", hover_color="#229647", font=ctk.CTkFont(weight="bold"), command=self.toggle_bot)
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.auto_live_btn = ctk.CTkButton(self.btn_frame, text="⚡ LIVE", fg_color="#3399ff", hover_color="#2277cc", font=ctk.CTkFont(weight="bold"), command=self.quick_start_live)
        self.auto_live_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # Mode Selection
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Execution Mode:", anchor="w")
        self.mode_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.mode_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["Recommendation Only", "Automatic Play"], variable=self.mode_var)
        self.mode_dropdown.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        
        # Model Selection. Default MUST match core/decision.py's active model (v20_preflopEq_AI) so
        # the UI label reflects what actually runs — the engine defaults to v20_preflopEq_AI
        # regardless, but a stale label here would be misleading (this dropdown's values/default
        # do NOT update themselves when core/decision.py's registry changes — bump both here too).
        self.model_var = ctk.StringVar(value="Herocules (v20_preflopEq_AI)")
        self.model_label = ctk.CTkLabel(self.sidebar, text="Decision Model:", anchor="w")
        self.model_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.sidebar,
            # Only models that actually load are listed (see core/decision.py registry). The
            # legacy Pluribus/v8-v11 entries were pruned — their weights are missing or use the
            # old 159-feature contract, so selecting them would output random actions live.
            # v20_preflopEq_AI/v20_preflopEq/v20/v19/v17_gauntlet/v17/v15/v14 = discretized
            # bet-size action space (raises-to-X / all-in); v13 kept as fallback.
            values=[
                "Herocules (v20_preflopEq_AI)",
                "Herocules (v20_preflopEq)",
                "Herocules (v20)",
                "Herocules (v19)",
                "Herocules (v17_gauntlet)",
                "Herocules (v17 Actor-Critic)",
                "Herocules (v15 DoN)",
                "Herocules (v14 Sized)",
                "Herocules (v13 Range-Aware)",
            ],
            variable=self.model_var,
            command=self.on_model_changed
        )
        self.model_dropdown.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        # Source Selection
        self.source_label = ctk.CTkLabel(self.sidebar, text="Input Source:", anchor="w")
        self.source_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
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
        self.source_dropdown.grid(row=8, column=0, padx=20, pady=5, sticky="ew")
        
        # Target Window (Only visible when Live Capture selected)
        self.window_label = ctk.CTkLabel(self.sidebar, text="Target Window Name:", anchor="w")
        self.window_combo = ctk.CTkComboBox(self.sidebar, values=["Bet365"], variable=self.target_window_var)
        
        # Big Blind Amount
        self.bb_label = ctk.CTkLabel(self.sidebar, text="Big Blind (in Cents):", anchor="w")
        self.bb_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.bb_entry = ctk.CTkEntry(self.sidebar, textvariable=self.big_blind_var)
        self.bb_entry.grid(row=10, column=0, padx=20, pady=5, sticky="ew")
        
        # Board Size Selection
        self.board_size_label = ctk.CTkLabel(self.sidebar, text="Board Size:", anchor="w")
        self.board_size_label.grid(row=11, column=0, padx=20, pady=(10, 0), sticky="w")
        self.board_size_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["6-Max", "10-Max"], variable=self.board_size_var)
        self.board_size_dropdown.grid(row=12, column=0, padx=20, pady=5, sticky="ew")
        
        # Looseness Slider Selection
        self.looseness_label = ctk.CTkLabel(self.sidebar, text="Pre-flop Looseness: +0%", anchor="w")
        self.looseness_label.grid(row=13, column=0, padx=20, pady=(10, 0), sticky="w")
        self.looseness_slider = ctk.CTkSlider(self.sidebar, from_=-0.20, to=0.20, number_of_steps=40, variable=self.looseness_var, command=self.update_looseness_label)
        self.looseness_slider.grid(row=14, column=0, padx=20, pady=5, sticky="ew")
        
        # NOTE: the preflop-chart/math-engine/bluff-engine toggle checkboxes were removed from the
        # UI (2026-07-15) -- they're no-ops for every currently loaded model (v13/v14/v15 all
        # bypass them; see core/decision.py's is_v13_model/is_sized_model guards), so the sidebar
        # showed live controls with no live effect. The underlying vars stay False (their pre-existing
        # default, "see true model actions") and are still threaded through make_decision() below for
        # forward compatibility with a future model that does use them.
        # State Machine indicator at bottom of sidebar
        self.state_frame = ctk.CTkFrame(self.sidebar, height=45, fg_color="#1e2129", corner_radius=8)
        self.state_frame.grid(row=15, column=0, padx=20, pady=20, sticky="ew")
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
        # 3 columns: table view (0,1) is weight=0 with an explicit minsize pinning it at its
        # previous rendered width (its own content minimum alone is much narrower, which would
        # otherwise shrink the board overview -- kept intentionally unchanged per earlier request).
        # The equity/action-distribution panel (2) is the only weighted column, so ALL surplus width
        # in this row (including whatever the sidebar frees up) flows into it instead.
        self.visual_frame.grid_columnconfigure((0, 1), weight=0, minsize=220)
        self.visual_frame.grid_columnconfigure(2, weight=1)
        self.visual_frame.grid_rowconfigure((0, 1), weight=1)

        # Title of telemetry
        self.telemetry_title = ctk.CTkLabel(self.visual_frame, text="Live Table Telemetry", font=ctk.CTkFont(size=15, weight="bold"))
        self.telemetry_title.grid(row=0, column=0, columnspan=3, pady=(10, 5))

        # Center Panel: Seating Table Layout (Visual Grid) spanning columns 0 and 1
        self.table_panel = ctk.CTkFrame(self.visual_frame, fg_color="#14161d", corner_radius=10)
        self.table_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
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
                
                self.comm_cards_val = ctk.CTkLabel(pot_frame, text="[--, --, --, --, --]", font=ctk.CTkFont(size=11))
                self.comm_cards_val.pack(pady=1)
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

                # "V ●" / "A ●" HUD dots -- labeled so the color coding (Blue/Green/Yellow/Red =
                # Tight->Maniac VPIP, Passive->Maniac AGG) is legible without prior context.
                lbl_vpip_tag = ctk.CTkLabel(vpip_agg_frame, text="V", font=ctk.CTkFont(size=7, weight="bold"), text_color="#5a6070")
                lbl_vpip_tag.pack(side="left", padx=(0, 1))
                lbl_vpip = ctk.CTkLabel(vpip_agg_frame, text="●", font=ctk.CTkFont(size=8), text_color="#1f222a")
                lbl_vpip.pack(side="left", padx=(0, 5))

                lbl_agg_tag = ctk.CTkLabel(vpip_agg_frame, text="A", font=ctk.CTkFont(size=7, weight="bold"), text_color="#5a6070")
                lbl_agg_tag.pack(side="left", padx=(0, 1))
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
                
                if key == 'hero':
                    self.hero_cards_val = ctk.CTkLabel(frame, text="[--, --]", font=ctk.CTkFont(size=12, weight="bold"), text_color="#2eb85c")
                    self.hero_cards_val.pack(pady=1)
        
        # Right Panel: Win Probability / Equity
        self.equity_panel = ctk.CTkFrame(self.visual_frame, fg_color="#1f222a", corner_radius=8)
        self.equity_panel.grid(row=1, column=2, padx=15, pady=15, sticky="nsew")
        self.equity_panel.grid_columnconfigure(0, weight=1)
        
        # Compact vertical rhythm through this whole panel (title/value/desc font sizes + pady) --
        # the panel's real height is fixed by the grid row, and content beyond it just gets clipped
        # (no scrolling), so trimming space here is what keeps ACTION DISTRIBUTION / MODEL CONTEXT
        # FEATURES below actually visible.
        # Three-up equity readout: Hand Win% (field-independent card quality, V20_preflopEq's
        # `hand_strength` feature) and Eq Edge (equity's edge over the field-size fair share,
        # V20_preflopEq's `equity_edge` feature) flank the main Board Eq (the range/board-aware
        # equity number that has always lived here, just renamed from "WIN PROBABILITY" -- that
        # name implied a plain showdown-probability, but this is specifically the range-aware
        # method's output). The two side stats are V20_preflopEq-only features; they read "-" for
        # any other active model (see update_equity_ui) rather than a misleading neutral number.
        self.equity_stats_row = ctk.CTkFrame(self.equity_panel, fg_color="transparent")
        self.equity_stats_row.pack(pady=(4, 0), fill="x")
        self.equity_stats_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.hand_strength_lbl_title = ctk.CTkLabel(self.equity_stats_row, text="HAND WIN%", font=ctk.CTkFont(weight="bold", size=10), text_color="#8a90a0")
        self.hand_strength_lbl_title.grid(row=0, column=0, sticky="n")
        self.hand_strength_val = ctk.CTkLabel(self.equity_stats_row, text="-", text_color="#c3c2b7", font=ctk.CTkFont(size=15, weight="bold"))
        self.hand_strength_val.grid(row=1, column=0, sticky="n")

        self.equity_lbl_title = ctk.CTkLabel(self.equity_stats_row, text="BOARD EQ", font=ctk.CTkFont(weight="bold", size=12))
        self.equity_lbl_title.grid(row=0, column=1, sticky="n")
        self.equity_val = ctk.CTkLabel(self.equity_stats_row, text="0.0%", text_color="#3399ff", font=ctk.CTkFont(size=22, weight="bold"))
        self.equity_val.grid(row=1, column=1, sticky="n")

        self.equity_edge_lbl_title = ctk.CTkLabel(self.equity_stats_row, text="EQ EDGE", font=ctk.CTkFont(weight="bold", size=10), text_color="#8a90a0")
        self.equity_edge_lbl_title.grid(row=0, column=2, sticky="n")
        self.equity_edge_val = ctk.CTkLabel(self.equity_stats_row, text="-", text_color="#c3c2b7", font=ctk.CTkFont(size=15, weight="bold"))
        self.equity_edge_val.grid(row=1, column=2, sticky="n")

        self.equity_desc = ctk.CTkLabel(self.equity_panel, text="W: 0.0%, D: 0.0%, L: 0.0%", font=ctk.CTkFont(size=11), text_color="#a0a0a0")
        self.equity_desc.pack(pady=0)

        # Range-aware opponent color breakdown (only populated when equity used the range-aware
        # method -- see update_equity_ui). Replaces what used to be a redundant raw-decimal
        # equity readout here (a display bug: parsing "Range-aware equity vs [...]: 0.71" for the
        # text after the colon threw away the useful color list and kept the equity number again).
        # Heights locked (see the reason/thinking labels above) so a varying opponent count doesn't
        # shift the panels below.
        self.equity_inpot_lbl = ctk.CTkLabel(self.equity_panel, text="", font=ctk.CTkFont(size=10), text_color="#a0a0a0", height=16)
        self.equity_inpot_lbl.pack(pady=0)

        self.equity_toact_lbl = ctk.CTkLabel(self.equity_panel, text="", font=ctk.CTkFont(size=10), text_color="#a0a0a0", height=16)
        self.equity_toact_lbl.pack(pady=(0, 4))

        # Recommended Action
        self.action_title_lbl = ctk.CTkLabel(self.equity_panel, text="RECOMMENDED ACTION", font=ctk.CTkFont(weight="bold", size=11), text_color="#8a90a0")
        self.action_title_lbl.pack(pady=(6, 0))

        self.action_val = ctk.CTkLabel(self.equity_panel, text="WAITING...", text_color="#ffd700", font=ctk.CTkFont(size=18, weight="bold"))
        self.action_val.pack(pady=2)

        # Locked height (2 lines' worth): text length varies turn to turn (a short "-" vs a full
        # policy dict / thinking sentence), and CTkLabel grows/shrinks to fit its wrapped line count
        # -- left unlocked, everything below (Action Distribution, Model Context Features) shifts
        # position every turn depending on how many lines this turn's text happened to wrap to.
        self.action_reason_lbl = ctk.CTkLabel(self.equity_panel, text="-", font=ctk.CTkFont(size=10), text_color="#8a90a0", wraplength=300, height=34)
        self.action_reason_lbl.pack(pady=1)

        self.thinking_lbl = ctk.CTkLabel(self.equity_panel, text="", font=ctk.CTkFont(size=13, weight="bold", slant="italic"), text_color="#5fa8d3", wraplength=300, height=48)
        self.thinking_lbl.pack(pady=(0, 1))

        # Action Distribution: overlaid raw (blue) vs temperature-sampled (gold) P(action) bars, one
        # per legal action this turn -- see _update_action_distribution/_redraw_dist_bar for the
        # blend logic. The action the sampler ("dice roll") actually picked gets a white bar outline
        # + gold ">" name/pct text. Row count covers the widest supported policy (V14/V15's 6-way
        # sized action space); unused rows are hidden for 3-way models (V13).
        self.ev_breakdown_frame = ctk.CTkFrame(self.equity_panel, fg_color="#181a20", corner_radius=6)
        self.ev_breakdown_frame.pack(fill="x", padx=10, pady=(8, 4))

        self.ev_title_lbl = ctk.CTkLabel(self.ev_breakdown_frame, text="ACTION DISTRIBUTION", font=ctk.CTkFont(weight="bold", size=10), text_color="#8a90a0")
        self.ev_title_lbl.pack(pady=(4, 0))

        self.dist_empty_lbl = ctk.CTkLabel(self.ev_breakdown_frame, text="Waiting for decision...", font=ctk.CTkFont(size=10), text_color="#8a90a0")
        self.dist_empty_lbl.pack(pady=(2, 6))

        # Locked height, sized for all 6 rows -- CTkFrame defaults to a fixed 200px height when
        # unset, and once grown to fit real rows it does NOT shrink back down when they're
        # grid_remove()'d (they don't shrink it), so letting height float produced a layout jump
        # between "waiting" and "populated" states (and a stale oversized gap after the first real
        # decision). Locking it to the real 6-row height keeps the panel a constant size always.
        self.dist_rows_container = ctk.CTkFrame(self.ev_breakdown_frame, fg_color="transparent", height=180)
        self.dist_rows_container.pack(fill="x", padx=8, pady=(0, 4))
        self.dist_rows_container.grid_columnconfigure(1, weight=1)

        self.action_dist_rows = []
        for i in range(6):  # max width across supported policies: V14/V15's 6-way sized actions
            name_lbl = ctk.CTkLabel(self.dist_rows_container, text="-", font=ctk.CTkFont(size=10, weight="bold"), width=58, anchor="w")
            name_lbl.grid(row=i, column=0, sticky="w", pady=1)

            # Raw tkinter Canvas (not a CTkProgressBar) so we can draw two overlaid, alpha-blended
            # bars: raw actor probability vs the post-temperature sampling probability.
            bar = tk.Canvas(self.dist_rows_container, height=14, bg="#3a3f4b", highlightthickness=0)
            bar.grid(row=i, column=1, sticky="ew", padx=6, pady=1)
            bar.bind("<Configure>", lambda e, idx=i: self._redraw_dist_bar(idx))

            pct_lbl = ctk.CTkLabel(self.dist_rows_container, text="0%", font=ctk.CTkFont(size=10), width=58, anchor="e")
            pct_lbl.grid(row=i, column=2, sticky="e", pady=1)

            ev_lbl = ctk.CTkLabel(self.dist_rows_container, text="", font=ctk.CTkFont(size=9), text_color="#8a90a0", width=48, anchor="e")
            ev_lbl.grid(row=i, column=3, sticky="e", pady=1)

            row = {'name': name_lbl, 'bar': bar, 'pct': pct_lbl, 'ev': ev_lbl, 'raw': 0.0, 'sampled': None, 'chosen': False}
            for w in (name_lbl, bar, pct_lbl, ev_lbl):
                w.grid_remove()  # hidden until the first real decision arrives
            self.action_dist_rows.append(row)
        
        # Context Features Frame
        self.context_tensor_frame = ctk.CTkFrame(self.equity_panel, fg_color="#181a20", corner_radius=6)
        self.context_tensor_frame.pack(fill="x", padx=10, pady=(3, 5))
        
        self.ctx_title_lbl = ctk.CTkLabel(self.context_tensor_frame, text="MODEL CONTEXT FEATURES", font=ctk.CTkFont(weight="bold", size=10), text_color="#8a90a0")
        self.ctx_title_lbl.pack(pady=(5, 2))
        
        self.ctx_row1_lbl = ctk.CTkLabel(self.context_tensor_frame, text="Pos: - | Stack: - BB | Pot: - BB | Eq: -%", font=ctk.CTkFont(size=11), text_color="#8a90a0")
        self.ctx_row1_lbl.pack(pady=1)
        
        self.ctx_row2_lbl = ctk.CTkLabel(self.context_tensor_frame, text="Odds: -% | Opps: - | Street: -", font=ctk.CTkFont(size=11), text_color="#8a90a0")
        self.ctx_row2_lbl.pack(pady=1)
        
        self.ctx_row3_lbl = ctk.CTkLabel(self.context_tensor_frame, text="VPIP: - | AGG: -", font=ctk.CTkFont(size=11), text_color="#8a90a0")
        self.ctx_row3_lbl.pack(pady=(1, 5))

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
            
            # Auto-select poker window if current is default or invalid
            valid_curr = any(curr == t for t in titles)
            if not valid_curr or curr == "Bet365" or curr == "":
                for t in titles:
                    if "NL Hold'em" in t or "Double Or Nothing" in t:
                        self.target_window_var.set(t)
                        self.append_log(f"[SYSTEM] Auto-selected poker window: {t}")
                        break
                        
        except Exception as e:
            self.append_log(f"[SYSTEM] Error refreshing window list: {e}")

    def update_looseness_label(self, val):
        self.looseness_label.configure(text=f"Pre-flop Looseness: {float(val):+.0%}")

    def on_source_changed(self, val):
        if val == "Live Capture":
            self.refresh_window_list()
            # show target window field in sidebar below looseness controls, shift state down
            self.window_label.grid(row=15, column=0, padx=20, pady=(10, 0), sticky="w")
            self.window_combo.grid(row=16, column=0, padx=20, pady=5, sticky="ew")
            self.state_frame.grid(row=17, column=0, padx=20, pady=20, sticky="ew")
            self.sidebar.grid_rowconfigure(18, weight=1) # set new spacer
        else:
            # hide target window field, restore state position
            self.window_label.grid_forget()
            self.window_combo.grid_forget()
            self.state_frame.grid(row=15, column=0, padx=20, pady=20, sticky="ew")
            self.sidebar.grid_rowconfigure(16, weight=1) # restore spacer

    def on_model_changed(self, val):
        self.decision_engine.set_active_model(val)
        self.append_log(f"[SYSTEM] Switched decision model to: {val}")

    def quick_start_live(self):
        self.append_log("[SYSTEM] Executing Auto-Live Quick Start...")
        
        # 1. Set mode to Automatic Play
        self.mode_var.set("Automatic Play")
        
        # 2. Set source to Live Capture
        self.source_var.set("Live Capture")
        self.on_source_changed("Live Capture") # Trigger layout shifts and window list refresh
        
        # 3. Search for the window matching the pattern
        try:
            # Refresh list of visible windows
            wins = get_visible_windows_with_pids()
            matched_title = None
            
            # Print search logs to guide the user
            self.append_log(f"[SYSTEM] Scanning {len(wins)} active windows...")
            
            # Filter windows that look like poker tables or clients
            poker_wins = []
            for pid, hwnd, title in wins:
                t_lower = title.lower()
                # Count various types of pipe/vertical bar characters (standard, box-drawing, full-width)
                pipes = title.count('|') + title.count('│') + title.count('｜')
                
                # Check for table keywords
                is_table = any(k in t_lower for k in ["hold'em", "omaha", "no limit", "pot limit", "nl", "pl", "ante", "double or nothing", "niveau", "table"])
                # Check for blind fraction like 50/100 or 0.10/0.20
                has_blind_fraction = bool(re.search(r'\d+/\d+', title))
                
                is_lobby = "bet365" in t_lower or "poker" in t_lower
                
                if is_table or has_blind_fraction or is_lobby or pipes > 0:
                    poker_wins.append({
                        'pid': pid,
                        'hwnd': hwnd,
                        'title': title,
                        'pipes': pipes,
                        'is_table': is_table or has_blind_fraction or pipes >= 3,
                        'is_lobby': is_lobby and not (is_table or has_blind_fraction)
                    })
            
            # Print found poker-related windows for debugging
            for w in poker_wins:
                self.append_log(f"  Candidate: '{w['title']}' (pipes: {w['pipes']}, table_match: {w['is_table']})")
                
            # Selection Strategy:
            # 1. First, search for a high-confidence TABLE window (is_table=True and pipes >= 2)
            for w in poker_wins:
                if w['is_table'] and w['pipes'] >= 2:
                    matched_title = f"[{w['pid']}] {w['title']}"
                    break
                    
            # 2. Second, search for any TABLE window
            if not matched_title:
                for w in poker_wins:
                    if w['is_table']:
                        matched_title = f"[{w['pid']}] {w['title']}"
                        break
                        
            # 3. Third, fallback to lobby window
            if not matched_title:
                for w in poker_wins:
                    if w['is_lobby']:
                        matched_title = f"[{w['pid']}] {w['title']}"
                        break
            
            if matched_title:
                self.target_window_var.set(matched_title)
                self.append_log(f"[SYSTEM] Auto-matched & selected target window: {matched_title}")
                
                # 4. Start the bot if not already running
                if not self.bot_running:
                    self.toggle_bot()
            else:
                self.append_log("[ERROR] No poker table or lobby window detected.")
                self.append_log("[SYSTEM] Listing all visible windows to debug:")
                list_wins = wins[:20]
                for pid, hwnd, title in list_wins:
                    self.append_log(f"  - [{pid}] '{title}' (pipes: {title.count('|')})")
                if len(wins) > 20:
                    self.append_log(f"  ... and {len(wins) - 20} more windows.")
        except Exception as e:
            self.append_log(f"[ERROR] Auto-Live detection failed: {e}")


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
            self.start_btn.configure(text="STOP BOT (F5)", fg_color="#dc3545", hover_color="#bd2130")
            self.append_log("[SYSTEM] Starting PHPHelp engine...")
            
            # Start background worker thread
            self.state_machine.start()
            self.bot_thread = threading.Thread(target=self.bot_worker_loop, daemon=True)
            self.bot_thread.start()
        else:
            # Stop the Bot
            self.bot_running = False
            self.start_btn.configure(text="START BOT (F5)", fg_color="#2eb85c", hover_color="#229647")
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
        self._awaiting_turn_clear = False   # fresh start -- don't inherit a stale gate from a prior run

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
                    self.last_window_title = f"Mock_{filename}"   # board id source for history
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
                            
                        # Parse blinds from window title if possible (e.g. "50/100")
                        try:
                            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                            if length > 0:
                                buf = ctypes.create_unicode_buffer(length + 1)
                                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                                win_title = buf.value
                                self.last_window_title = win_title   # board id source for history
                                # Try matching decimal stakes first: e.g. "0.10/0.20" or "€0.10/€0.20"
                                blind_match = re.search(r'(?:€|\$|£)?(\d+\.\d+)/(?:€|\$|£)?(\d+\.\d+)', win_title)
                                if blind_match:
                                    sb = float(blind_match.group(1)) * 100.0
                                    bb = float(blind_match.group(2)) * 100.0
                                else:
                                    # Fallback to integer stakes: e.g. "50/100" or "10/20"
                                    blind_match = re.search(r'(?:€|\$|£)?(\d+)/(?:€|\$|£)?(\d+)', win_title)
                                    if blind_match:
                                        sb = float(blind_match.group(1))
                                        bb = float(blind_match.group(2))
                                        
                                if blind_match:
                                    if self.big_blind_var.get() != bb:
                                        self.big_blind_var.set(bb)
                                        self.append_log(f"[SYSTEM] Auto-parsed blinds from window title: SB={sb:.0f}, BB={bb:.0f}")
                        except Exception as e_title:
                            self.append_log(f"[SYSTEM] Error parsing blinds from window title: {e_title}")
                            
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
                raw_state = self.vision.read_board_state(img, board_size=self.board_size_var.get())
                
                # Check for hand reset
                if self.table_state.detect_hand_reset(raw_state):
                    self.append_log("[SYSTEM] New hand detected. Resetting table state history.")
                    self.table_state.reset()
                    
                    if not source.startswith("Mock:"):
                        try:
                            baseline_stacks, hero_name, dealer_name = self.xml_tracker.get_baseline_stacks()
                            if baseline_stacks:
                                self.append_log(f"[XML Tracker] Loaded baseline stacks: {baseline_stacks} (Hero: {hero_name}, Dealer: {dealer_name})")
                                self.pending_baseline_stacks = (baseline_stacks, hero_name, dealer_name)
                        except Exception as e:
                            self.append_log(f"[XML Tracker] Error loading baseline stacks: {e}")
                            self.pending_baseline_stacks = None
                    else:
                        self.pending_baseline_stacks = None
                    
                # Update stabilized table state datamodel
                self.table_state.update(raw_state)
                
                # Apply baseline stacks seeding on the first frame of a new hand (only for Live Capture)
                if not source.startswith("Mock:") and self.pending_baseline_stacks:
                    baseline_stacks, hero_name, dealer_name = self.pending_baseline_stacks
                    self.table_state.seed_stacks(baseline_stacks, hero_name, dealer_name)
                    self.pending_baseline_stacks = None
                    
                stabilized_state = self.table_state.to_dict()
                
                # Update GUI visual elements continuously
                self.after(0, self.update_telemetry_ui, stabilized_state)
                
                # Check if it's Hero's turn (look for active buttons and verify Hero has cards dealt)
                button_matches = self.vision.match_templates_in_roi(
                    img, self.vision.rois['buttons'], self.vision.button_templates, threshold=0.85, max_matches=1
                )
                
                if not button_matches:
                    # Not our turn.
                    # If we were in a mid-state (e.g. DECIDING), we safely reset to WAITING_FOR_TURN
                    if self.state_machine.state not in ['IDLE', 'WAITING_FOR_TURN']:
                        self.state_machine.error_occurred()

                    # Confirmed the client has actually moved off our last action -- safe to act
                    # again next time buttons appear (see _awaiting_turn_clear in __init__).
                    self._awaiting_turn_clear = False

                    # Keep the last real decision's equity/action/distribution on screen for a
                    # minimum hold time instead of blanking it the instant it's not Hero's turn --
                    # this loop iterates every ~1s, so without this the display reset almost
                    # immediately after each decision, giving no real time to read it.
                    if time.time() - self.last_decision_ts >= self.MIN_DECISION_DISPLAY_SECONDS:
                        self.after(0, self.update_action_ui, "WAITING...", "Not Hero's turn", 0)

                    # Sleep before next continuous tracking frame
                    time.sleep(1.0 if not source.startswith("Mock:") else 5.0)
                    continue

                # Buttons are visible, but if we haven't yet seen a "not our turn" frame since our
                # last action, this is almost certainly the SAME stale buttons still on screen from
                # before (client hasn't visually caught up yet) -- do NOT treat it as a fresh turn.
                if self._awaiting_turn_clear:
                    time.sleep(1.0 if not source.startswith("Mock:") else 2.0)
                    continue

                # It IS our turn!
                if self.state_machine.state != 'WAITING_FOR_TURN':
                    time.sleep(0.5)
                    continue

                if self.state_machine.state == 'WAITING_FOR_TURN':
                    fold_btn_coord = button_matches[0][1] # relative coordinates
                    self.state_machine.turn_detected()
                    self.append_log("\n--- HERO TURN DETECTED ---")
                    
                    active_opps_by_seat = {k: v for k, v in stabilized_state['opponents'].items() if v.get('is_active', True)}
                    active_opps = list(active_opps_by_seat.values())

                    # --- MAX THREAT VPIP/AGG AGGREGATION ---
                    # Aligned with Bet365/iPoker HUD color thresholds:
                    # VPIP: Tight (Blue: <18%), Normal (Green: 18-26%), Loose (Yellow: 26-35%), Maniac (Red: >35%)
                    # AGG: Passive (Blue: <36%), Normal (Green: 36-56%), Aggressive (Yellow: 56-71%), Maniac (Red: >71%)
                    vpip_map = {'Red': 0.45, 'Yellow': 0.30, 'Green': 0.22, 'Blue': 0.10}
                    agg_map = {'Red': 0.85, 'Yellow': 0.63, 'Green': 0.46, 'Blue': 0.18}
                    max_vpip = 0.0
                    max_agg = 0.0
                    
                    if active_opps:
                        for opp in active_opps:
                            v_col = opp.get('vpip_color')
                            a_col = opp.get('agg_color')
                            
                            if v_col and v_col in vpip_map:
                                v_val = vpip_map[v_col]
                                if v_val > max_vpip: max_vpip = v_val
                                
                            if a_col and a_col in agg_map:
                                a_val = agg_map[a_col]
                                if a_val > max_agg: max_agg = a_val
                            
                    if max_vpip == 0.0: max_vpip = 0.3
                    if max_agg == 0.0: max_agg = 0.4
                    
                    stabilized_state['opp_vpip_norm'] = max_vpip
                    stabilized_state['opp_agg_norm'] = max_agg
                    # ---------------------------------------
                    
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
                    else:
                        num_opponents = self.opponents_var.get()
                    num_sims = self.simulations_var.get()
                    
                    # V13 requires RANGE-AWARE equity (hero equity vs each opponent's VPIP-color
                    # range) to match how it was trained — using vs-random equity here would be a
                    # silent train/serve mismatch. Falls back to vs-random if v13 isn't active or
                    # the range-aware calc can't run.
                    equity = None
                    sim_msg = None
                    equity_meta = {"method": "vs-random", "opp_colors": None, "num_opponents": num_opponents}
                    # V13/V14/V15/V17/V17_gauntlet/V19/V20 were all trained with RANGE-AWARE
                    # equity (hero equity vs each opponent's VPIP-color range); feeding vs-random
                    # here would be a silent train/serve mismatch. Use the matching version's
                    # identical impl.
                    _active_lower = self.decision_engine.active_model_name.lower()
                    if 'v20_preflopeq_ai' in _active_lower or 'v20_preflopeq' in _active_lower or 'v20' in _active_lower or 'v19' in _active_lower or 'v17' in _active_lower or 'v15' in _active_lower or 'v14' in _active_lower or 'v13' in _active_lower:
                        try:
                            # NOTE: resolution order matters -- 'v20_preflopeq' is a substring of
                            # 'v20_preflopeq_ai', and 'v20' is a substring of both, so each must be
                            # checked before the shorter name it contains or the naive ordering
                            # would always match the wrong (earlier-fix-less) branch.
                            if 'v20_preflopeq_ai' in _active_lower:
                                from versions.v20_preflopEq_AI.self_play.simulator import compute_range_aware_equity
                            elif 'v20_preflopeq' in _active_lower:
                                from versions.v20_preflopEq.self_play.simulator import compute_range_aware_equity
                            elif 'v20' in _active_lower:
                                from versions.v20.self_play.simulator import compute_range_aware_equity
                            elif 'v19' in _active_lower:
                                from versions.v19.self_play.simulator import compute_range_aware_equity
                            elif 'v17_gauntlet' in _active_lower:
                                from versions.v17_gauntlet.self_play.simulator import compute_range_aware_equity
                            elif 'v17' in _active_lower:
                                from versions.v17.self_play.simulator import compute_range_aware_equity
                            elif 'v15' in _active_lower:
                                from versions.v15.self_play.simulator import compute_range_aware_equity
                            elif 'v14' in _active_lower:
                                from versions.v14.self_play.simulator import compute_range_aware_equity
                            else:
                                from versions.v13.self_play.simulator import compute_range_aware_equity
                            # [V20_preflopEq Finding 1] An opponent whose HUD color hasn't been
                            # classified yet (vpip_color is None) is a real, demonstrably-contesting
                            # seat -- NOT the same as them not being there. Map unknown -> 'Yellow'
                            # (this codebase's existing "no info" convention elsewhere, e.g.
                            # opponents_profiles.get(...).get('vpip', 0.3) -> Yellow's band) rather
                            # than silently dropping them. See versions/v20_preflopEq/SPECS.md
                            # Finding 1 -- "full range"/omission is actually the most hero-favorable
                            # (least conservative) wrong answer, not a cautious middle ground.
                            opp_colors = [o.get('vpip_color') or 'Yellow' for o in active_opps]
                            equity_meta["opp_colors"] = opp_colors
                            # For v20_preflopEq this breakdown now ALSO drives the equity call
                            # below (front_colors=colors_in_pot); for every other model it stays
                            # display-only (see _classify_opponents_by_action_order's docstring).
                            colors_in_pot, colors_still_to_act = self._classify_opponents_by_action_order(
                                stabilized_state, active_opps_by_seat
                            )
                            equity_meta["opp_colors_in_pot"] = colors_in_pot
                            equity_meta["opp_colors_still_to_act"] = colors_still_to_act
                            # sims=250 (2026-07-16, live-serving only -- default is 150, matching
                            # what training uses): at 150 sims, 2*SE on a ~50% equity estimate is
                            # ~8pp, comparable to a color band's own width (e.g. Yellow=[26,35], 9pp
                            # wide) -- too noisy to reliably resolve band-driven differences. 250
                            # brings 2*SE down to ~6.3pp. Doesn't touch training's own sims=150 call
                            # (versions/*/self_play/simulator.py) or its default -- same formula
                            # either way, just less noise around the same expected value live.
                            if 'v20_preflopeq' in _active_lower and colors_in_pot is not None:
                                # [V20_preflopEq Finding 2] front (already acted this round,
                                # guaranteed in -- no VPIP fold-roll) vs after (still to act,
                                # normal roll), using the SAME positional classifier already
                                # computed above for display -- now actually driving the equity
                                # math too, matching training's fix exactly. Falls back to the
                                # flat legacy call below if the dealer button wasn't detected this
                                # frame (classifier returns (None, None) when it can't establish
                                # order, distinct from a real empty front/after list).
                                ra = compute_range_aware_equity(
                                    stabilized_state['hero_cards'],
                                    stabilized_state['community_cards'],
                                    colors_still_to_act,
                                    sims=250,
                                    front_colors=colors_in_pot,
                                )
                            else:
                                ra = compute_range_aware_equity(
                                    stabilized_state['hero_cards'],
                                    stabilized_state['community_cards'],
                                    opp_colors,
                                    sims=250,
                                )
                            if ra is not None:
                                equity = ra
                                equity_meta["method"] = "range-aware"
                                sim_msg = f"Range-aware equity vs {opp_colors or 'random'}: {equity:.2f}"
                            else:
                                equity_meta["fallback_reason"] = "range-aware returned None (no HUD colors?)"
                        except Exception as e:
                            equity_meta["fallback_reason"] = f"range-aware raised: {e}"
                            self.append_log(f"[Equity] range-aware failed ({e}); vs-random fallback")

                    # [V20_preflopEq] hand_strength: field-independent card-quality signal, only
                    # meaningful (and only computed, to avoid a needless live MC call for every
                    # other model) when v20_preflopEq or v20_preflopEq_AI is active (identical
                    # feature, both versions' own contract module). Preflop: O(1) lookup.
                    # Postflop: a cheap vs-1-random MC call, same recipe as simulator.py's own
                    # _hand_strength.
                    hand_strength = 0.5
                    if 'v20_preflopeq' in _active_lower:
                        try:
                            if 'v20_preflopeq_ai' in _active_lower:
                                from versions.v20_preflopEq_AI.core.contract import preflop_hand_strength
                            else:
                                from versions.v20_preflopEq.core.contract import preflop_hand_strength
                            _hero_cards = stabilized_state['hero_cards']
                            _community = stabilized_state['community_cards']
                            if len(_community) == 0:
                                hand_strength = preflop_hand_strength(_hero_cards[0], _hero_cards[1])
                            else:
                                hand_strength, _ = self.evaluator.calculate_equity(
                                    _community, _hero_cards, num_opponents=1, num_simulations=200
                                )
                        except Exception as e:
                            self.append_log(f"[Equity] hand_strength computation failed ({e}); using neutral 0.5")
                    equity_meta["hand_strength"] = hand_strength

                    if equity is None:
                        equity, sim_msg = self.evaluator.calculate_equity(
                            stabilized_state['community_cards'],
                            stabilized_state['hero_cards'],
                            num_opponents=num_opponents,
                            num_simulations=num_sims
                        )
                    equity_meta["value"] = equity
                    # [V20_preflopEq] equity_edge: equity's edge over the field-size fair share
                    # (equity*(num_active+1), 1.0 = exactly average for this field size) --
                    # display-only here; contract.py derives its own copy internally from
                    # state.equity + the active-opponent count for the actual model input, this is
                    # just that same formula computed for the dashboard, against the FINAL equity
                    # (post range-aware/vs-random fallback) and the same opponent count `equity`
                    # was itself computed against (detected_opponents / num_opponents above).
                    equity_meta["equity_edge"] = equity * (num_opponents + 1)
                    self.last_equity_meta = equity_meta

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

                stabilized_state['big_blind'] = self.big_blind_var.get()
                
                board_state = self.table_state.to_board_state(
                    call_amount=call_amount,
                    equity=equity,
                    big_blind=self.big_blind_var.get()
                )
                # [V20_preflopEq] hand_strength computed earlier alongside equity (see
                # equity_meta["hand_strength"]) -- to_board_state() doesn't take it as a
                # constructor param (BoardState field is additive/optional, see core/board_state.py),
                # so set it directly here, same as `equity` itself is threaded through above.
                board_state.hand_strength = self.last_equity_meta.get("hand_strength", 0.5) if self.last_equity_meta else 0.5

                decision_tuple = self.decision_engine.make_decision(
                    board_state,
                    use_preflop_chart=self.layer_preflop_var.get(),
                    use_math_engine=self.layer_math_var.get(),
                    use_bluff_engine=self.layer_bluff_var.get(),
                    use_dynamic_sizing=self.layer_sizing_var.get(),
                    bet_raise_available=bet_raise_available,
                    check_call_available=check_call_available
                )
                
                action = decision_tuple[0]
                reason = decision_tuple[1]
                bet_size = decision_tuple[2]
                ev_dict = decision_tuple[3] if len(decision_tuple) > 3 else None
                
                # Save parsed states for debug / diagnostics
                self.last_table_state = stabilized_state
                self.last_equity = equity
                self.last_decision = (action, reason, bet_size)
                self.last_ev_dict = ev_dict   # full model output (policy + Q-values + decision path)
                
                # Safeguard: If decided action is CHECK, but we detected the middle button as KALD/CALL or it's unavailable,
                # override to FOLD to prevent accidental calling.
                if action == 'CHECK' and not source.startswith("Mock:"):
                    if not check_call_available or any(w in text_upper for w in ["KALD", "CALL", "KLD", "KND"]):
                        self.append_log("[Safeguard] WARNING: Decided CHECK but Check/Call button is KALD or unavailable! Overriding to FOLD.")
                        action = 'FOLD'
                        reason = f"Safeguard: Blocked accidental call on CHECK decision. (OCR: '{cc_text}', Avail: {check_call_available})"
                        if ev_dict is not None:
                            ev_dict['thinking'] = "Thinking: Check/Call unavailable -- folding rather than risk an accidental call."
                        # Re-save decision with safeguard applied
                        self.last_decision = (action, reason, bet_size)
                
                # Record Hero's own action in the table state action history
                if action == 'FOLD':
                    self.table_state.action_history.append('f')
                elif action in ['CHECK', 'CALL']:
                    self.table_state.action_history.append('c')
                elif action.startswith('BET') or action.startswith('RAISE'):
                    self.table_state.action_history.append('r')
                
                self.append_log(f"[Decision] DECIDED BRANCH: **{action}**")
                self.append_log(f"[Decision] Reason: {reason}")
                _thinking = (ev_dict or {}).get('thinking')
                if _thinking:
                    self.append_log(f"[Decision] {_thinking}")
                if bet_size > 0:
                    self.append_log(f"[Decision] Size Allocation: {bet_size} units")
                    
                self.after(0, self.update_action_ui, action, reason, bet_size, ev_dict)
                self.last_decision_ts = time.time()   # starts the min-display-time hold (see __init__)
                self._record_turn_history()   # append this decided turn to history/<board_id>/turns.jsonl
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
                        window_size=win_size,
                        log_fn=self.append_log
                    )
                    if success:
                        self.append_log("[Automation] Thread interrupt completed successfully.")
                    else:
                        self.append_log("[Automation] Thread interrupt failed.")
                    # We just physically clicked -- don't act again until a frame confirms the
                    # client actually moved on (see _awaiting_turn_clear in __init__/the turn-check
                    # above). Not set in Recommendation Only mode: nothing was clicked there, so
                    # gating on a "turn clear" that a human elsewhere controls could stall future
                    # recommendations indefinitely.
                    self._awaiting_turn_clear = True
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
                self.after(0, lambda: self.start_btn.configure(text="START BOT (F5)", fg_color="#2eb85c", hover_color="#229647"))
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
        self.hero_cards_val.configure(text=f"{state['hero_cards']}")
        self.comm_cards_val.configure(text=f"{state['community_cards']}")
        self.pot_val.configure(text=f"{state['pot_size']}")
        
        # 1. Update Hero Seat
        hero_widget = self.seat_widgets['hero']
        hero_stack = state['hero_stack']
        is_hero_dealer = (state.get('dealer_idx', -1) == 0)
        hero_text = "Hero [D]" if is_hero_dealer else "Hero"
        
        if hero_stack > 0:
            hero_widget['name'].configure(text=hero_text)
            hero_widget['stack'].configure(text=f"{hero_stack} chips")
            hero_widget['frame'].configure(fg_color="#1b4d3e") # Active green background
            hero_widget['name'].configure(text_color="#2eb85c")
            hero_widget['stack'].configure(text_color="#2eb85c")
        else:
            hero_widget['name'].configure(text=hero_text)
            hero_widget['stack'].configure(text="0")
            hero_widget['frame'].configure(fg_color="#2d3038") # Folded
            hero_widget['name'].configure(text_color="#8a90a0")
            hero_widget['stack'].configure(text_color="#8a90a0")
            
        if is_hero_dealer:
            hero_widget['frame'].configure(border_color="#ffd700", border_width=2)
        else:
            hero_widget['frame'].configure(border_width=0)
            
        # Update Hero VPIP and AGG HUD indicator colors on the dashboard
        color_map = {
            'Blue': "#3399ff",
            'Green': "#2eb85c",
            'Yellow': "#ffd700",
            'Red': "#e55353",
            None: "#1f222a" # Hidden
        }
        hero_vpip_c = state.get('hero_vpip_color')
        hero_agg_c = state.get('hero_agg_color')
        if 'vpip' in hero_widget and 'agg' in hero_widget:
            hero_widget['vpip'].configure(text_color=color_map.get(hero_vpip_c, "#1f222a"))
            hero_widget['agg'].configure(text_color=color_map.get(hero_agg_c, "#1f222a"))
            
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
                is_dealer = (state.get('dealer_idx', -1) == i)
                opp_text = f"[D] {name}" if is_dealer else name
                
                widget['name'].configure(text=opp_text)
                
                if is_dealer:
                    widget['frame'].configure(border_color="#ffd700", border_width=2)
                else:
                    widget['frame'].configure(border_width=0)
                
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

    def update_equity_ui(self, equity, sim_msg=None):
        self.equity_val.configure(text=f"{equity * 100:.1f}%")
        # Color based on equity strength
        if equity > 0.65:
            self.equity_val.configure(text_color="#2eb85c") # Green (Very strong)
        elif equity > 0.45:
            self.equity_val.configure(text_color="#3399ff") # Blue (Medium)
        else:
            self.equity_val.configure(text_color="#e55353") # Red (Weak)
            
        # self.last_equity_meta is set synchronously (worker thread) right before this call gets
        # scheduled via self.after, so it's always the meta for THIS equity value.
        meta = self.last_equity_meta or {}

        # [V20_preflopEq] Hand Win% / Eq Edge side stats -- only meaningful for this model (every
        # other model neither computes hand_strength nor trains with equity_edge), so show "-"
        # rather than a misleading number when a different model is active.
        if 'v20_preflopeq' in self.decision_engine.active_model_name.lower():
            hs = meta.get("hand_strength")
            self.hand_strength_val.configure(text=f"{hs * 100:.1f}%" if hs is not None else "-")
            edge = meta.get("equity_edge")
            self.equity_edge_val.configure(text=f"{edge:.2f}x" if edge is not None else "-")
        else:
            self.hand_strength_val.configure(text="-")
            self.equity_edge_val.configure(text="-")

        if meta.get("method") == "range-aware":
            # Range-aware equity is a single MC number, not a W/D/L split -- the old code path
            # below (splitting sim_msg on ":") was written for the vs-random evaluator's message
            # and, for this method, just re-displayed the equity as a redundant raw decimal
            # ("Range-aware equity vs [...]: 0.71" -> "0.71") while throwing away the useful
            # opponent color list. Show that breakdown instead.
            self.equity_desc.configure(text="Range-aware equity")
            in_pot = meta.get("opp_colors_in_pot")
            still_to_act = meta.get("opp_colors_still_to_act")
            if in_pot is None and still_to_act is None:
                self.equity_inpot_lbl.configure(text="")
                self.equity_toact_lbl.configure(text="(no dealer button detected this frame)")
            else:
                self.equity_inpot_lbl.configure(text=f"In pot: [{', '.join(in_pot) if in_pot else '-'}]")
                self.equity_toact_lbl.configure(text=f"Still to act: [{', '.join(still_to_act) if still_to_act else '-'}]")
        else:
            self.equity_inpot_lbl.configure(text="")
            self.equity_toact_lbl.configure(text="")
            if sim_msg:
                # We already format sim_msg correctly in evaluator.py
                # Expected format from evaluator: Simulated 2000 hands: W=45.0%, D=5.0%, L=50.0%
                # Just extract the part after the colon
                parts = sim_msg.split(":", 1)
                desc_text = parts[1].strip() if len(parts) > 1 else sim_msg
                self.equity_desc.configure(text=desc_text)

    def update_action_ui(self, action, reason, bet_size, ev_dict=None):
        # Format action text nicely (bet_size is chip-fraction math -> round for display, e.g.
        # avoid "26.400000000000002")
        text = action
        if bet_size > 0:
            text = f"{action} ({bet_size:.0f})"
            
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
        if "Pluribus Q-Net" in clean_reason:
            if "Raw ->" in clean_reason:
                clean_reason = clean_reason.split("Raw ->")[-1].replace(")", "").strip()
                clean_reason = clean_reason.replace(",", " | ").replace(":", ": ")
        elif ":" in clean_reason:
            clean_reason = clean_reason.split(":")[-1].strip()
        self.action_reason_lbl.configure(text=clean_reason)

        thinking = (ev_dict or {}).get('thinking')
        self.thinking_lbl.configure(text=thinking or "")

        # Update the Action Distribution bars (P(action) per legal action + the sampled pick)
        self._update_action_distribution(ev_dict)


        # Update Context Tensors display
        if hasattr(self, 'last_table_state') and self.last_table_state:
            try:
                state = self.last_table_state
                big_blind = state.get('big_blind', 25.0)
                
                pos_val = state.get('position', 0)
                
                hero_stack = state.get('hero_stack', 0)
                stack_bb = hero_stack / big_blind
                
                pot_size = state.get('pot_size', 0)
                pot_bb = pot_size / big_blind
                
                eq_val = self.last_equity * 100.0 if hasattr(self, 'last_equity') else 0.0
                
                call_amount = state.get('call_amount', 0)
                pot_odds = (call_amount / (pot_size + call_amount)) * 100.0 if (pot_size + call_amount) > 0 else 0.0
                
                num_opps = self.opponents_var.get()
                
                board = state.get('community_cards', [])
                board_len = len(board)
                if board_len == 0:
                    street_name = "Pre-flop"
                elif board_len == 3:
                    street_name = "Flop"
                elif board_len == 4:
                    street_name = "Turn"
                else:
                    street_name = "River"
                    
                self.ctx_row1_lbl.configure(text=f"Pos: {pos_val} | Stack: {stack_bb:.1f} BB | Pot: {pot_bb:.1f} BB | Eq: {eq_val:.1f}%")
                self.ctx_row2_lbl.configure(text=f"Odds: {pot_odds:.1f}% | Opps: {num_opps} | Str: {street_name}")
                
                opp_vpip = state.get('opp_vpip_norm', 0.3)
                opp_agg = state.get('opp_agg_norm', 0.4)
                self.ctx_row3_lbl.configure(text=f"VPIP: {opp_vpip:.2f} | AGG: {opp_agg:.2f}")
            except Exception as e:
                pass

    def _update_action_distribution(self, ev_dict):
        """Render P(action), one row per legal action this turn, as an overlaid bar: the actor's
        RAW probability (blue) vs the same distribution after live temperature-sharpening (gold) --
        `core/decision.py` SAMPLES the executed action from the sharpened distribution, not the raw
        one, so the two can genuinely disagree (sharpening pulls weight off already-unlikely
        actions toward the favorite). Where both bars cover the same ground we blend the colors so
        the overlap reads as agreement; a solid tail in either color shows which distribution
        the temperature scaling pushed that action's weight *from* or *to*.

        `ev_dict` keys directly ARE the raw action probabilities (see core/decision.py:
        `ev_dict = evs.copy()`); ACTION_DIAG_ORDER picks those out from the diagnostic/bookkeeping
        keys (decision_path, thinking, q_vals, sampled_probs, ...) mixed into the same dict.
        `chosen_key` is the raw policy bucket the sampler picked -- it can differ from the final
        executed `action` (e.g. a sized raise gets translated to RAISE_SLIDER_x, or a safeguard
        overrides it), so highlighting it shows what the model's dice actually rolled.
        """
        # Force the row container back to its locked height on every call. CTkFrame only ever
        # grows to fit content and never shrinks back on its own -- and dist_empty_lbl must stay
        # PACKED at all times (never pack_forget()'d) because forget-then-pack() re-inserts it at
        # the END of the pack stack, after dist_rows_container, which silently reordered it below
        # the (locked-height, now-empty-looking) row container the first time this toggled.
        self.dist_rows_container.configure(height=180)
        rows = self.action_dist_rows
        keys = [k for k in ACTION_DIAG_ORDER if k in (ev_dict or {})]

        if not keys:
            for row in rows:
                for w in (row['name'], row['bar'], row['pct'], row['ev']):
                    w.grid_remove()
            self.dist_empty_lbl.configure(text="Waiting for decision..." if not ev_dict else "(no action distribution for this model)")
            return

        self.dist_empty_lbl.configure(text="")
        q_vals = ev_dict.get('q_vals') or {}
        sampled_all = ev_dict.get('sampled_probs')   # None for the legacy argmax path
        chosen_key = ev_dict.get('chosen_key')

        for i, row in enumerate(rows):
            if i >= len(keys):
                for w in (row['name'], row['bar'], row['pct'], row['ev']):
                    w.grid_remove()
                continue

            k = keys[i]
            raw = max(0.0, min(1.0, float(ev_dict.get(k) or 0.0)))
            samp = max(0.0, min(1.0, float(sampled_all[k]))) if sampled_all and k in sampled_all else None
            is_chosen = (k == chosen_key)
            text_color = "#ffd700" if is_chosen else "#c9cdd6"

            row['name'].configure(text=(">" if is_chosen else " ") + ACTION_DISPLAY_NAMES.get(k, k), text_color=text_color)
            if samp is None:
                row['pct'].configure(text=f"{raw*100:.0f}%", text_color=text_color)
            else:
                row['pct'].configure(text=f"{raw*100:.0f}%→{samp*100:.0f}%", text_color=text_color)
            ev_val = q_vals.get(k)
            row['ev'].configure(text=f"{ev_val:+.2f}bb" if ev_val is not None else "")

            row['raw'], row['sampled'], row['chosen'] = raw, samp, is_chosen
            for w in (row['name'], row['bar'], row['pct'], row['ev']):
                w.grid()
            self._redraw_dist_bar(i)

    def _redraw_dist_bar(self, idx):
        """Draw one Action Distribution row's overlaid raw/sampled bar onto its Canvas. Split into
        [0, min] blended, (min, max] in the color of whichever distribution is larger there -- so a
        solid blue tail means temperature-sharpening REDUCED that action's share, a solid gold tail
        means it INCREASED it, and pure blend means the two agree. Bound to <Configure> (not just
        called on data updates) so it redraws correctly once the canvas gets its real pixel width
        from the grid manager, and again if the window is ever resized."""
        row = self.action_dist_rows[idx]
        canvas = row['bar']
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        canvas.delete("all")
        if w <= 1 or h <= 1:
            return

        raw, samp = row.get('raw', 0.0), row.get('sampled')
        if samp is None:
            bw = int(w * raw)
            if bw > 0:
                canvas.create_rectangle(0, 0, bw, h, fill=_rgb_hex(_DIST_RAW_RGB), outline="")
        else:
            lo, hi = min(raw, samp), max(raw, samp)
            lo_px, hi_px = int(w * lo), int(w * hi)
            if lo_px > 0:
                canvas.create_rectangle(0, 0, lo_px, h, fill=_blend_hex(_DIST_RAW_RGB, _DIST_SAMPLED_RGB), outline="")
            if hi_px > lo_px:
                owner = _DIST_RAW_RGB if raw > samp else _DIST_SAMPLED_RGB
                canvas.create_rectangle(lo_px, 0, hi_px, h, fill=_rgb_hex(owner), outline="")

        if row.get('chosen'):
            canvas.create_rectangle(0, 0, w - 1, h - 1, outline="#ffffff", width=1)

    def poll_keyboard_shortcuts(self):
        try:
            import ctypes
            
            # Virtual key code for ESC is 0x1B
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                if self.bot_running:
                    self.bot_running = False
                    self.after(0, lambda: self.start_btn.configure(text="START BOT (F5)", fg_color="#2eb85c", hover_color="#229647"))
                    self.append_log("[SYSTEM] Emergency Abort: Escape key pressed. Stopping bot.")
                    self.state_machine.stop()
                    
            # Virtual key code for F5 is 0x74 -> toggle live bot ON/OFF (edge-triggered).
            if ctypes.windll.user32.GetAsyncKeyState(0x74) & 0x8000:
                if not getattr(self, '_f5_was_down', False):
                    self._f5_was_down = True
                    self.toggle_bot()
            else:
                self._f5_was_down = False

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

    def _board_id_from_title(self, title):
        """Derive a STABLE, filesystem-safe board id from the window title. A single match/table
        keeps one long numeric id across the whole game (e.g. 1170780915), while the rest of the
        title changes hand-to-hand — stakes (0.10), tournament LEVEL (Niveau 1 -> Niveau 2), and
        client version (v.26.1). Those must NOT fork the folder, so the id is anchored on that long
        number, prefixed with the stable game-type words that precede it for readability, e.g.
        'Double Or Nothing 0.10 1170780915 NL Holdem Niveau 2 v.26.1' -> 'Double_Or_Nothing_1170780915'."""
        if not title:
            return "unknown"
        # Long digit run (>=6) = the table/tournament id. Blinds/levels/versions are <=4-5 digits,
        # so the real id is unambiguously the longest such run.
        nums = re.findall(r'\d{6,}', title)
        if nums:
            match_id = max(nums, key=len)
            prefix = re.sub(r'[^A-Za-z]+', '_', re.match(r'\D*', title).group(0)).strip('_')[:30]
            return f"{prefix}_{match_id}".strip('_') if prefix else match_id
        # Fallback (mock images / clients with no id): strip stakes, then sanitize the whole title.
        s = re.sub(r'(?:€|\$|£)?\d+(?:\.\d+)?\s*/\s*(?:€|\$|£)?\d+(?:\.\d+)?', '', title)
        s = re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
        return (s[:60] or "table")

    def _ensure_history_session(self):
        """Point the recorder at history/<board_id>/ for the current window title, starting a new
        session folder whenever the board id changes. Returns the session dir (or None on failure)."""
        try:
            board_id = self._board_id_from_title(self.last_window_title)
            if board_id != self.session_board_id:
                self.session_board_id = board_id
                self.session_history_dir = os.path.join("history", board_id)
                os.makedirs(os.path.join(self.session_history_dir, "flagged"), exist_ok=True)
                self.session_turn_count = 0
                self.append_log(f"[History] Recording turns -> {self.session_history_dir}/turns.jsonl")
            return self.session_history_dir
        except Exception as e:
            self.append_log(f"[History] Could not open session dir: {e}")
            return None

    def _decode_model_input(self, ev):
        """Decode the ACTUAL scalars the model consumed from its recorded input tensor (final step
        of the 35-dim ContractV12 context). Ground truth — unlike re-deriving from the raw vision
        state, which can diverge (a bridge bug). Indices per versions/v13/core/contract.py.
        Returns {} if no tensor was captured."""
        try:
            last = (ev or {}).get("model_input", {}).get("ctx")[0][-1]
            street = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}.get(round(last[6] * 3.0), "?")
            return {
                "position": round(last[0] * 5.0, 2),
                "hero_stack_bb": round(last[1] * 400.0, 1),
                "pot_bb": round(last[2] * 1000.0, 2),
                "equity": round(last[3], 3),
                "pot_odds": round(last[4], 3),
                "num_active": round(last[5] * 10.0),
                "street": street,
                "to_call_bb": round(last[9] * 400.0, 2),
            }
        except Exception:
            return {}

    def _classify_opponents_by_action_order(self, stabilized_state, active_opps_by_seat):
        """Best-effort split of active opponents' HUD colors into 'in pot' (already acted this
        street) vs 'still to act' (haven't acted yet this street) -- purely from seat position +
        dealer button + street, since vision/table_state don't track per-seat action status
        directly. NOT aware of reopened action (a check-raise means an earlier seat must act
        again, which this can't detect) -- a positional approximation, not ground truth.

        [V20_preflopEq] When that model is active, this split now ALSO drives the actual equity
        call (front -> compute_range_aware_equity's `front_colors`, guaranteed in, no VPIP roll;
        still_to_act -> the legacy `opp_colors`, normal roll) -- see the call site above. For
        every OTHER model, this remains display-only: their compute_range_aware_equity has no
        front/after concept at all -- preflop it gives every opponent a flat VPIP-weighted chance
        of even being in the hand, postflop it treats every still-active opponent as fully in.
        Returns (colors_in_pot, colors_still_to_act), or (None, None) if the dealer button wasn't
        detected this frame (can't establish an order without it).
        """
        dealer_idx = stabilized_state.get('dealer_idx', -1)
        if dealer_idx not in range(0, 6):
            return None, None

        button_seat = 'hero' if dealer_idx == 0 else f'seat_{dealer_idx}'
        if button_seat not in SEAT_ORDER_CLOCKWISE:
            return None, None
        start = SEAT_ORDER_CLOCKWISE.index(button_seat)
        order = SEAT_ORDER_CLOCKWISE[start:] + SEAT_ORDER_CLOCKWISE[:start]  # button first, then clockwise

        # First-to-act this street: postflop = seat right after the button; preflop = 2 seats
        # after that (past the blinds) -- a full-ring approximation; doesn't adjust if the blinds
        # themselves already folded.
        is_preflop = len(stabilized_state.get('community_cards', [])) == 0
        first_to_act_offset = 3 if is_preflop else 1
        order = order[first_to_act_offset:] + order[:first_to_act_offset]

        if 'hero' not in order:
            return None, None
        hero_pos = order.index('hero')
        before_hero = order[:hero_pos]        # already had their turn this street -> "in pot"
        after_hero = order[hero_pos + 1:]     # haven't acted yet this street -> "still to act"

        def colors_for(seat_list):
            colors = []
            for seat_key in seat_list:
                opp = active_opps_by_seat.get(seat_key)
                if opp:
                    # [V20_preflopEq Finding 1] Same unknown->Yellow mapping as the main opp_colors
                    # list above -- a seat with no HUD color read yet is still really there.
                    colors.append(opp.get('vpip_color') or 'Yellow')
            return colors

        return colors_for(before_hero), colors_for(after_hero)

    def _curate_opponents(self, state):
        """Per-seat opponent snapshot for the board-state layer (objective read from vision)."""
        opps = []
        for seat_key, opp in (state.get("opponents") or {}).items():
            if not isinstance(opp, dict):
                continue
            opps.append({
                "seat": seat_key,
                "is_active": opp.get("is_active", True),
                "vpip_color": opp.get("vpip_color"),
                "agg_color": opp.get("agg_color"),
                "stack": opp.get("stack"),
            })
        return opps

    def _build_turn_record(self):
        """Two-layer, replay-ready snapshot of the latest turn (shared by the recorder + F12):
          board_state -> the OBJECTIVE table read; collect these across turns = the match array.
          evaluation  -> the MODEL's read of that state (equity, actor policy, critic Q, input tensors).
          action      -> the decision taken from the two layers above.
        """
        state = self.last_table_state or {}
        ev = self.last_ev_dict or {}
        em = self.last_equity_meta or {}
        action, reason, bet_size = (self.last_decision or ("?", "?", 0.0))

        bb = float(state.get("big_blind") or 0) or None
        def _bb(x):
            try:
                return round(float(x) / bb, 2) if bb else None
            except Exception:
                return None
        board = state.get("community_cards") or []
        street = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}.get(len(board), f"{len(board)}cards")
        pot = state.get("pot_size")
        # to_call is NOT stored in stabilized_state (it's computed downstream in the decision loop
        # and passed straight to the board_state), so state.get('call_amount') was always None ->
        # to_call/to_call_bb/pot_odds logged as null. Source the AUTHORITATIVE price the model
        # actually consumed from its input tensor (ctx[9]*400 = to_call BB; ctx[4] = pot_odds),
        # falling back to the raw state only if no tensor was captured.
        seen = self._decode_model_input(ev)
        to_call_bb = seen.get("to_call_bb")
        if to_call_bb is not None:
            to_call = round(to_call_bb * bb, 2) if bb else None
        else:
            to_call = state.get("call_amount")
            to_call_bb = _bb(to_call)
        pot_odds = seen.get("pot_odds")
        if pot_odds is None:
            try:
                denom = float(pot or 0) + float(to_call or 0)
                pot_odds = round(float(to_call or 0) / denom, 3) if denom > 0 else 0.0
            except Exception:
                pot_odds = None

        return {
            "format": 2,
            "turn": self.session_turn_count,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "board_id": self.session_board_id,
            "window_title": self.last_window_title,
            "flagged": False,
            # LAYER 1 — objective board state. Accumulate board_state across a match's turns = the match.
            "board_state": {
                "street": street,
                "hero_cards": state.get("hero_cards"),
                "board": board,
                "hero_position": state.get("hero_position"),
                "hero_stack": state.get("hero_stack"),
                "hero_stack_bb": _bb(state.get("hero_stack")),
                "pot": pot,
                "pot_bb": _bb(pot),
                "to_call": to_call,
                "to_call_bb": to_call_bb,
                "pot_odds": pot_odds,
                "big_blind": state.get("big_blind"),
                "num_opponents": em.get("num_opponents"),
                "opponents": self._curate_opponents(state),
            },
            # LAYER 2 — the model's evaluation of that board state.
            "evaluation": {
                "model": getattr(self.decision_engine, "active_model_name", None),
                "equity": self.last_equity,
                "equity_method": em.get("method"),
                "equity_opp_colors": em.get("opp_colors"),
                # [V20_preflopEq] front/after split (only meaningfully populated when this model is
                # active -- see _classify_opponents_by_action_order; None for every other model,
                # same as before) and the two new engineered features. Previously shown live in the
                # dashboard but never persisted, so a past session couldn't be audited after the
                # fact -- now part of the permanent turn record.
                "equity_opp_colors_in_pot": em.get("opp_colors_in_pot"),
                "equity_opp_colors_still_to_act": em.get("opp_colors_still_to_act"),
                "hand_strength": em.get("hand_strength"),
                "equity_edge": em.get("equity_edge"),
                "actor_policy": {k: ev.get(k) for k in ACTION_DIAG_ORDER if k in ev},
                "critic_q": ev.get("q_vals"),
                "model_input": ev.get("model_input"),   # exact input tensors -> faithful replay
            },
            # The decision derived from the two layers above.
            "action": {"chosen": action, "bet_size": bet_size, "reason": reason},
        }

    def _record_turn_history(self):
        """Append the just-decided turn to history/<board_id>/turns.jsonl. Runs every live turn."""
        if self.last_ev_dict is None and self.last_decision is None:
            return
        hist_dir = self._ensure_history_session()
        if not hist_dir:
            return
        try:
            self.session_turn_count += 1
            record = self._build_turn_record()
            with open(os.path.join(hist_dir, "turns.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            self.append_log(f"[History] Failed to record turn: {e}")

    def save_diagnostics(self):
        """F12: MARK the current turn as worth investigating. Saves the heavy artifacts (screenshot
        + layered summary) under history/<board_id>/flagged/ and records a pointer in flags.jsonl,
        so it's a bookmark into the continuous turn history rather than a separate capture."""
        if self.last_raw_img is None:
            self.append_log("[SYSTEM] Warning: No active turn data to save.")
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Anchor the flag inside the current board's history session (fallback to diagnostics/).
            hist_dir = self._ensure_history_session()
            if hist_dir:
                dir_name = os.path.join(hist_dir, "flagged", f"turn_{self.session_turn_count}_{timestamp}")
            else:
                dir_name = os.path.join("diagnostics", f"turn_{timestamp}")
            os.makedirs(dir_name, exist_ok=True)

            # Record the flag pointer so replay/analysis knows which turns were marked.
            if hist_dir:
                try:
                    with open(os.path.join(hist_dir, "flags.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"turn": self.session_turn_count, "ts": timestamp,
                                            "dir": dir_name, "action": (self.last_decision or ["?"])[0]},
                                           default=str) + "\n")
                except Exception:
                    pass

            # Save screenshot
            screenshot_path = os.path.join(dir_name, "screenshot.png")
            cv2.imwrite(screenshot_path, self.last_raw_img)

            # ---- Gather the full decision trace: INPUT (perception) -> FEATURES -> MODEL OUTPUT ----
            state = self.last_table_state or {}
            ev = self.last_ev_dict or {}
            eqm = self.last_equity_meta or {}
            # Ground truth: what the model ACTUALLY consumed (decoded from its input tensor). Preferred
            # over re-deriving from the raw vision state, which can diverge (a bridge bug) and mislead.
            seen = self._decode_model_input(ev)
            action, reason, bet_size = (self.last_decision or ("?", "?", 0.0))
            policy = {k: ev.get(k) for k in ACTION_DIAG_ORDER if k in ev}   # actor
            q_vals = ev.get("q_vals")                                                # critic EV/action
            decision_path = ev.get("decision_path")

            # Reconstruct an APPROXIMATE labeled 9D vector from the raw vision state. NOTE: this is
            # NOT authoritative — `seen` (decoded from the actual input tensor) is what the model got.
            ctx_labeled = {}
            context_vector = None
            try:
                big_blind = float(state.get('big_blind', 25.0)) or 25.0
                position = float(state.get('hero_position', 0)) / 10.0
                bankroll = (float(state.get('hero_stack', 0)) / big_blind) / 500.0
                pot = (float(state.get('pot_size', 0)) / big_blind) / 500.0
                equity = float(self.last_equity)
                call_amount = float(state.get('call_amount', 0))
                pot_size = float(state.get('pot_size', 0))
                pot_odds = (call_amount / (pot_size + call_amount)) if (pot_size + call_amount) > 0 else 0.0
                num_opponents = float(self.opponents_var.get()) / 10.0
                board_len = len(state.get('community_cards', []))
                street_level = {0: 0.0, 3: 1.0, 4: 2.0}.get(board_len, 3.0) / 3.0
                opp_vpip_norm = state.get('opp_vpip_norm', 0.3)
                opp_agg_norm = state.get('opp_agg_norm', 0.4)
                context_vector = [position, bankroll, pot, equity, pot_odds, num_opponents, street_level, opp_vpip_norm, opp_agg_norm]
                ctx_labeled = {"position": position, "bankroll": bankroll, "pot": pot, "equity": equity,
                               "pot_odds": pot_odds, "num_opponents": num_opponents, "street_level": street_level,
                               "opp_vpip_norm": opp_vpip_norm, "opp_agg_norm": opp_agg_norm}
            except Exception:
                pass

            diag_data = {
                "timestamp": timestamp,
                "chosen_action": action, "bet_size": bet_size, "reason": reason,
                "actor_policy": policy, "critic_q_vals": q_vals,
                "equity": self.last_equity, "equity_meta": eqm,
                "context_vector": context_vector, "context_labeled": ctx_labeled,
                "model_seen": seen,   # authoritative: decoded from the actual model input tensor
                "decision_path": decision_path, "table_state": self.last_table_state,
                "hand_history_len": len(getattr(self.decision_engine, 'hand_history_buffer', []) or []),
            }
            with open(os.path.join(dir_name, "telemetry.json"), "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=4, default=str)

            # ---- Human-readable, layered summary: localizes WHERE a bad decision came from ----
            street_name = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}.get(len(state.get('community_cards', [])), "?")
            bb = float(state.get('big_blind', 25.0)) or 25.0
            def _bb(x):
                try: return f"{float(x)/bb:.1f}BB"
                except Exception: return "?"
            def _dist(d, fmt):
                if not d: return "  (n/a — not an actor-critic model)"
                ks = [k for k in ACTION_DIAG_ORDER if k in d]
                return "  " + "  ".join(f"{k} {format(d.get(k, float('nan')), fmt)}" for k in ks)

            L = []
            L.append(f"=== TURN DIAGNOSTIC — {timestamp} ===")
            L.append(f"Model : {self.decision_engine.active_model_name}")
            L.append(f"CHOSE : {action}   bet={bet_size}   reason={reason}")
            L.append("")
            L.append("--- LAYER 1: PERCEPTION  (RAW vision read — cross-check vs screenshot.png) ---")
            L.append(f"  Hero cards : {state.get('hero_cards')}")
            L.append(f"  Board      : {state.get('community_cards')}   ({street_name})")
            L.append(f"  Position   : {state.get('hero_position')}")
            L.append(f"  Hero stack : {state.get('hero_stack')}  ({_bb(state.get('hero_stack'))})")
            L.append(f"  Pot        : {state.get('pot_size')}  ({_bb(state.get('pot_size'))})")
            L.append(f"  To call    : {state.get('call_amount')}  ({_bb(state.get('call_amount'))})")
            L.append(f"  Opp colors : {eqm.get('opp_colors')}   (num_opponents={eqm.get('num_opponents')})")
            L.append("  >> If any disagree with the screenshot -> OCR / PARSE bug (Layer 1).")
            L.append("")
            L.append("--- LAYER 2: FEATURES  (what the model ACTUALLY consumed — decoded from its input tensor) ---")
            if seen:
                L.append(f"  street     : {seen.get('street')}     equity : {seen.get('equity')}   [method: {eqm.get('method')}]")
                L.append(f"  pot        : {seen.get('pot_bb')}BB    to_call : {seen.get('to_call_bb')}BB    pot_odds : {seen.get('pot_odds')}")
                L.append(f"  hero_stack : {seen.get('hero_stack_bb')}BB    position : {seen.get('position')}    num_active : {seen.get('num_active')}")
                # Bridge check: does what the model consumed match the raw vision read?
                raw_street = {0: 'Preflop', 3: 'Flop', 4: 'Turn', 5: 'River'}.get(len(state.get('community_cards', [])), '?')
                mism = []
                if seen.get('street') != raw_street:
                    mism.append(f"street model={seen.get('street')} vs OCR={raw_street}")
                rc = state.get('call_amount')
                if rc is None and seen.get('to_call_bb'):
                    mism.append(f"to_call OCR=None but model saw {seen.get('to_call_bb')}BB (bridge filled it)")
                if mism:
                    L.append(f"  (!) MODEL-INPUT vs RAW-OCR MISMATCH -> BRIDGE issue: {mism}")
            else:
                L.append(f"  (no input tensor captured; approx from vision) equity {self.last_equity}, pot_odds {ctx_labeled.get('pot_odds')}")
            if eqm.get("fallback_reason"):
                L.append(f"  (!) equity fell back to vs-random: {eqm.get('fallback_reason')}")
            L.append("  >> These are the model's TRUE inputs. Wrong equity -> range/color bug;")
            L.append("     wrong price/street here but right in Layer 1 -> BRIDGE bug (not OCR, not the model).")
            L.append("")
            L.append("--- LAYER 3: POLICY  (given correct inputs, did the model choose right?) ---")
            L.append("  Actor policy P(action):")
            L.append(_dist(policy, ".2f"))
            L.append("  Critic Q (EV vs fold, ~BB):")
            L.append(_dist(q_vals, "+.2f"))
            L.append(f"  Chosen action : {action}")
            if policy and q_vals:
                try:
                    pol_pick, q_pick = max(policy, key=policy.get), max(q_vals, key=q_vals.get)
                    if pol_pick != q_pick:
                        L.append(f"  (!) ACTOR/CRITIC DISAGREE: actor prefers {pol_pick}, critic values {q_pick} highest")
                        L.append("      -> policy possibly miscalibrated vs the model's own value estimate.")
                except Exception:
                    pass
            L.append("  >> If Layers 1-2 are correct but this action is wrong -> MODEL / POLICY issue (Layer 3).")
            L.append("")
            L.append("Files: screenshot.png | telemetry.json | logs.txt | expected.txt")
            with open(os.path.join(dir_name, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(L))

            # Template so the user can label the intended action -> each flag becomes a case we can study.
            with open(os.path.join(dir_name, "expected.txt"), "w", encoding="utf-8") as f:
                f.write("EXPECTED ACTION (FOLD/CALL/RAISE/ALLIN): \nWHY (what did the model miss?): \n")

            # Save recent log history
            with open(os.path.join(dir_name, "logs.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self.recent_logs))

            self.append_log(f"[SYSTEM] Turn #{self.session_turn_count} FLAGGED -> {dir_name}/ (open summary.txt)")
        except Exception as e:
            self.append_log(f"[ERROR] Failed to save diagnostics: {e}")

if __name__ == "__main__":
    app = PHPHelpApp()
    app.mainloop()
