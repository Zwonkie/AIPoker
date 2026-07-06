import os
import glob
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

class XMLTracker:
    def __init__(self, history_dir=None):
        # Default path based on user's local directory
        if history_dir is None:
            self.history_dir = r"C:\Users\zwonk\AppData\Local\Poker at Bet365.DK\data\Zwonkie\History\Data\Tournaments"
        else:
            self.history_dir = history_dir

    def get_latest_xml_file(self) -> str:
        """Finds the most recently modified XML file in the history directory."""
        if not os.path.exists(self.history_dir):
            logger.warning(f"XML history directory does not exist: {self.history_dir}")
            return None

        xml_files = glob.glob(os.path.join(self.history_dir, "*.xml"))
        if not xml_files:
            return None

        # Sort files by last modification time in descending order
        xml_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return xml_files[0]

    def clean_number(self, num_str: str) -> float:
        """Helper to parse numeric values from XML attributes, removing commas."""
        if not num_str:
            return 0.0
        try:
            # Remove commas and convert to float (or int if possible)
            cleaned = num_str.replace(",", "").strip()
            val = float(cleaned)
            return int(val) if val.is_integer() else val
        except ValueError:
            return 0.0

    def parse_latest_hand_stacks(self, file_path: str) -> tuple:
        """
        Parses the given XML file and returns:
        - dictionary: {player_name: ending_stack}
        - string: hero_name (from the session general section)
        """
        if not file_path or not os.path.exists(file_path):
            return {}, ""

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 1. Get Hero Nickname
            hero_name = ""
            general_section = root.find("general")
            if general_section is not None:
                nickname_elem = general_section.find("nickname")
                if nickname_elem is not None and nickname_elem.text:
                    hero_name = nickname_elem.text.strip()

            # If nickname is not found, fallback to "Zwonkie"
            if not hero_name:
                hero_name = "Zwonkie"

            # 2. Get Last Game block
            games = root.findall("game")
            if not games:
                return {}, hero_name

            last_game = games[-1]
            players_elem = last_game.find("general/players")
            if players_elem is None:
                return {}, hero_name

            ending_stacks = {}
            for player in players_elem.findall("player"):
                name = player.get("name")
                if not name:
                    continue

                starting_chips = self.clean_number(player.get("chips", "0"))
                bet = self.clean_number(player.get("bet", "0"))
                win = self.clean_number(player.get("win", "0"))

                # End stack calculation: starting chips - total bet + total won
                ending_stack = max(0.0, starting_chips - bet + win)
                ending_stacks[name] = ending_stack

            return ending_stacks, hero_name

        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            return {}, ""

    def get_baseline_stacks(self) -> tuple:
        """Scans the directory, parses the latest XML, and returns (ending_stacks, hero_name)."""
        latest_file = self.get_latest_xml_file()
        if not latest_file:
            return {}, ""
        return self.parse_latest_hand_stacks(latest_file)
