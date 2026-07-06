import unittest
import os
import shutil
import tempfile
from core.xml_tracker import XMLTracker
from core.table_state import TableState

class TestXMLTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for XML files
        self.test_dir = tempfile.mkdtemp()
        self.tracker = XMLTracker(history_dir=self.test_dir)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_get_latest_xml_file(self):
        # 1. Empty directory returns None
        self.assertIsNone(self.tracker.get_latest_xml_file())

        # 2. Add an old XML file
        file1 = os.path.join(self.test_dir, "file1.xml")
        with open(file1, "w") as f:
            f.write("old file")
        os.utime(file1, (1000000000, 1000000000))  # Set modification time in past

        # 3. Add a newer XML file
        file2 = os.path.join(self.test_dir, "file2.xml")
        with open(file2, "w") as f:
            f.write("new file")
        os.utime(file2, (2000000000, 2000000000))

        # Check latest file
        latest = self.tracker.get_latest_xml_file()
        self.assertEqual(os.path.basename(latest), "file2.xml")

    def test_parse_latest_hand_stacks(self):
        # Create a mock xml file representing a tournament hand history
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<session sessioncode="8666181219">
 <general>
  <nickname>Zwonkie</nickname>
  <tournamentcode>1169880146</tournamentcode>
 </general>
 <game gamecode="12356203209">
  <general>
   <players>
    <player name="JustHere4TheRB" chips="1,500" bet="0" win="0"/>
    <player name="StigdeBearsky" chips="1,500" bet="20" win="0"/>
    <player name="Jambolad" chips="1,500" bet="710" win="1,440"/>
    <player name="Zwonkie" chips="1,500" bet="710" win="0"/>
   </players>
  </general>
 </game>
</session>
"""
        xml_file = os.path.join(self.test_dir, "tourney.xml")
        with open(xml_file, "w", encoding="utf-8") as f:
            f.write(xml_content)

        stacks, hero = self.tracker.parse_latest_hand_stacks(xml_file)

        self.assertEqual(hero, "Zwonkie")
        # JustHere4TheRB: 1500 - 0 + 0 = 1500
        self.assertEqual(stacks.get("JustHere4TheRB"), 1500)
        # StigdeBearsky: 1500 - 20 + 0 = 1480
        self.assertEqual(stacks.get("StigdeBearsky"), 1480)
        # Jambolad: 1500 - 710 + 1440 = 2230
        self.assertEqual(stacks.get("Jambolad"), 2230)
        # Zwonkie: 1500 - 710 + 0 = 790
        self.assertEqual(stacks.get("Zwonkie"), 790)

    def test_table_state_fuzzy_seeding(self):
        # Initialize table state
        table_state = TableState()
        table_state.reset()

        # Simulate first OCR frame populating names (with slight OCR errors/truncation)
        table_state.opponents = {
            'seat_1': {'name': 'JustHere4TheR', 'stack': 0, 'state': 'Active', 'is_active': True},
            'seat_2': {'name': 'StigdeBear', 'stack': 0, 'state': 'Active', 'is_active': True},
            'seat_3': {'name': 'Jambolad', 'stack': 0, 'state': 'Active', 'is_active': True},
        }

        # Stacks from XML (correct spellings)
        xml_stacks = {
            'JustHere4TheRB': 1500,
            'StigdeBearsky': 1480,
            'Jambolad': 2230,
            'Zwonkie': 790
        }

        # Seed stacks!
        table_state.seed_stacks(xml_stacks, hero_name="Zwonkie")

        # Assert Hero stack was seeded directly
        self.assertEqual(table_state.hero_stack, 790)

        # Assert opponents were seeded via fuzzy matching and names auto-corrected
        self.assertEqual(table_state.opponents['seat_1']['name'], 'JustHere4TheRB')
        self.assertEqual(table_state.opponents['seat_1']['stack'], 1500)

        self.assertEqual(table_state.opponents['seat_2']['name'], 'StigdeBearsky')
        self.assertEqual(table_state.opponents['seat_2']['stack'], 1480)

        self.assertEqual(table_state.opponents['seat_3']['name'], 'Jambolad')
        self.assertEqual(table_state.opponents['seat_3']['stack'], 2230)

if __name__ == "__main__":
    unittest.main()
