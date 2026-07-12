import json
import os
import tarfile
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated, Optional

SLASH = "\\" if os.name == "nt" else "/"
STAGES = ["preflop", "flop", "turn", "river", "showdown"]

# Annotated types for validation
Timestamp = Annotated[int, Field(ge=100000000, le=999999999)]
Stage = Annotated[str, Field(pattern=r"^[0-9]+\/[0-9]+$")]
Card = Annotated[str, Field(pattern=r"^[1-9,TJQKA][schd]$")]
Action = Annotated[str, Field(pattern=r"^[BfkbcrAQK-]+$")]

class HdbRecord(BaseModel):
    timestamp: Timestamp
    dealer: int
    hand_num: int
    num_players: int
    flop: Stage
    turn: Stage
    river: Stage
    showdown: Stage
    card1: Optional[Card] = None
    card2: Optional[Card] = None
    card3: Optional[Card] = None
    card4: Optional[Card] = None
    card5: Optional[Card] = None

    def __init__(self, *args):
        try:
            super().__init__(**dict(zip(self.model_fields.keys(), args)))
        except ValidationError:
            raise

    @property
    def cards(self):
        cards = [self.card1, self.card2, self.card3, self.card4, self.card5]
        return [c for c in cards if c is not None]

    @property
    def pots(self):
        pots = []
        for stage in STAGES[1:]:
            n, s = getattr(self, stage).split("/")
            pots.append({"num_players": int(n), "stage": stage[0], "size": int(s)})
        return pots

class HrosterRecord(BaseModel):
    timestamp: Timestamp
    num_players: int
    players: list[str]

    def __init__(self, *args):
        try:
            super().__init__(timestamp=args[0], num_players=args[1], players=args[2:])
        except ValidationError:
            raise

class PdbRecord(BaseModel):
    player: str
    timestamp: Timestamp
    num_players: int
    position: int
    preflop: Action
    flop: Action
    turn: Action
    river: Action
    bankroll: int
    total_bet: int
    total_win: int
    card1: Optional[Card] = None
    card2: Optional[Card] = None

    def __init__(self, *args):
        try:
            super().__init__(**dict(zip(self.model_fields.keys(), args)))
        except ValidationError:
            raise

    @property
    def cards(self):
        cards = [self.card1, self.card2]
        return [c for c in cards if c is not None]

    @property
    def bets(self):
        return [{"actions": getattr(self, stage), "stage": stage[0]} for stage in STAGES[:-1]]


class IRCTierParser:
    TIERS = {
        "holdem": "tier1",
        "holdem1": "tier1",
        "holdem2": "tier2",
        "holdem3": "tier3",
    }

    def __init__(self, fname_in: str, out_dir: str):
        self.fname_in = fname_in
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.tier_limits = {"tier1": 25000, "tier2": 25000, "tier3": 25000}
        self.out_files = {}
        self.tier_counts = {}
        for tier in set(self.TIERS.values()):
            self.out_files[tier] = open(os.path.join(self.out_dir, f"hands_{tier}.jsonl"), "w")
            self.tier_counts[tier] = 0

    def __del__(self):
        for f in self.out_files.values():
            f.close()

    def parse(self):
        split = lambda x: x.decode().strip().split()
        
        with tarfile.open(self.fname_in, "r:gz") as tar_in:
            for member in tar_in.getmembers():
                if not member.name.endswith(".tgz"):
                    continue
                
                fname_group = member.name.rsplit(SLASH, 1)[-1].rsplit('/', 1)[-1]
                game_type = fname_group.split(".", 1)[0]
                
                if game_type not in self.TIERS:
                    continue
                
                tier = self.TIERS[game_type]
                
                if self.tier_counts[tier] >= self.tier_limits[tier]:
                    if all(self.tier_counts[t] >= self.tier_limits[t] for t in self.tier_counts):
                        print("All tiers reached hand limit. Stopping.")
                        return
                    continue
                    
                print(f"Extracting {fname_group} for {tier} (Target: {self.tier_limits[tier]} hands)...")
                
                file_group = tar_in.extractfile(member)
                if not file_group:
                    continue
                    
                with tarfile.open(fileobj=file_group, mode="r:gz") as tar_group:
                    folder_group = fname_group.rstrip(".tgz").replace(".", SLASH)
                    
                    # Read HDB
                    fname_hdb = next((m for m in tar_group.getmembers() if m.name.endswith('hdb')), None)
                    if not fname_hdb:
                        continue
                    file_hdb = tar_group.extractfile(fname_hdb)
                    iter_hdb = iter(file_hdb)
                    
                    # Read HRoster
                    fname_hroster = next((m for m in tar_group.getmembers() if m.name.endswith('hroster')), None)
                    if not fname_hroster:
                        continue
                    file_hroster = tar_group.extractfile(fname_hroster)
                    iter_hroster = iter(file_hroster)
                    
                    # Read PDBs
                    file_pdb = {}
                    iter_pdb = {}
                    for fname in tar_group.getnames():
                        if "/pdb/pdb." in fname or "\\pdb\\pdb." in fname:
                            player = fname.split("pdb.")[-1]
                            file_pdb[player] = tar_group.extractfile(fname)
                            if file_pdb[player]:
                                iter_pdb[player] = iter(file_pdb[player])
                    
                    # Process
                    try:
                        pdb = {}
                        for k, v in iter_pdb.items():
                            try:
                                pdb[k] = PdbRecord(*[s for s in split(next(v)) if s])
                            except StopIteration:
                                pass
                                
                        while True:
                            try:
                                hdb = HdbRecord(*[s for s in split(next(iter_hdb)) if s])
                                while True:
                                    hroster = HrosterRecord(*[s for s in split(next(iter_hroster)) if s])
                                    if hroster.timestamp >= hdb.timestamp:
                                        break
                                
                                if hdb.timestamp < hroster.timestamp:
                                    continue
                                
                                _id = f"{folder_group}_{hdb.timestamp}"
                                
                                pdb_curr = {}
                                pdb_missing = False
                                for player in hroster.players:
                                    if player not in pdb:
                                        pdb_missing = True
                                        break
                                    while pdb[player].timestamp < hdb.timestamp:
                                        try:
                                            pdb[player] = PdbRecord(*[s for s in split(next(iter_pdb[player])) if s])
                                        except StopIteration:
                                            break
                                    if pdb[player].timestamp > hdb.timestamp:
                                        pdb_missing = True
                                        break
                                    pdb_curr[player] = pdb[player]
                                
                                if pdb_missing or len(hroster.players) != len(pdb_curr):
                                    continue
                                
                                hand = {
                                    "_id": _id,
                                    "board": hdb.cards,
                                    "dealer": hdb.dealer,
                                    "game": game_type,
                                    "hand_num": hdb.hand_num,
                                    "num_players": hdb.num_players,
                                    "players": {
                                        k: {
                                            "total_bet": v.total_bet,
                                            "bankroll": v.bankroll,
                                            "bets": v.bets,
                                            "pocket_cards": v.cards,
                                            "position": v.position,
                                            "total_win": v.total_win,
                                        }
                                        for k, v in pdb_curr.items()
                                    },
                                    "pots": hdb.pots,
                                }
                                
                                self.out_files[tier].write(json.dumps(hand) + "\n")
                                self.tier_counts[tier] += 1
                                if self.tier_counts[tier] >= self.tier_limits[tier]:
                                    print(f"Tier {tier} reached limit of {self.tier_limits[tier]} hands.")
                                    break
                                
                            except ValidationError:
                                continue
                    except StopIteration:
                        pass
                    except Exception as e:
                        print(f"Error parsing group: {e}")

if __name__ == "__main__":
    parser = IRCTierParser(
        fname_in="tools/data/raw/IRCdata.tgz",
        out_dir="tools/data/parsed"
    )
    parser.parse()
    print("Done parsing.")
