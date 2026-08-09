from card_utils import card_data, Card
import json
import httpx
from pathlib import Path

CARDS = [
    "Knight", "Giant", "Archer", "Goblins", "Pekka", "MiniPekka",
    "Minions", "Skeletons", "SkeletonArmy", "Balloon", "Witch",
    "Barbarians", "Golem", "Valkyrie", "Bomber", "Musketeer",
    "BabyDragon", "Prince", "Wizard", "SpearGoblins",
    "GiantSkeleton", "HogRider", "MinionHorde","RoyalGiant",
    "Princess", "ThreeMusketeers", "BlowdartGoblin", "AngryBarbarians",
    "Bats", "DartBarrell", "RoyalHogs", "Cannon", "Xbow",
    "IceWizard", "SkeletonWarriors", "DarkPrince", "LavaHound",
    "IceSpirits", "FireSpirits", "Miner", "ZapMachine", "Bowler",
    "Rage", "RageBarbarian", "BattleRam", "Fireball", "Arrows"
]

english_names = [card_data[each]['englishName'] for each in CARDS]
resolved = dict(zip(english_names, CARDS))
with open("cards.json") as f:
    cards = json.load(f)["items"]

names = {each['name']: each["iconUrls"]["medium"] for each in cards}
for each in english_names:
    img = httpx.get(names[each]).content
    Path(f"images/{each}.png").write_bytes(img)
    print(f"Downloaded {each}")
