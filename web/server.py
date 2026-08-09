"""
Browser-playable host for the simulator - no terminal, no local Python needed for players,
just open a URL. Two players connecting get auto-paired into a match; the server runs the
actual battle.py engine headlessly and streams state over a WebSocket.

Run: uvicorn server:app --host 0.0.0.0 --port 8000   (from this directory)
Then open http://<host>:8000 in a browser. Share the same URL with a second person (or a
second browser tab) to get paired into a match.
"""
SIMULATOR_VERSION = "v0.1.1-tower-range-fix"
import asyncio
import json
import os
import sys
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'clasher_new')
sys.path.insert(0, SIM_DIR)
os.chdir(SIM_DIR)  # card_utils.py opens gamedata.json etc. relative to cwd

import battle
import player as player_mod
from core import Position
from card_utils import card_data

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SIM_DIR, 'client_side', 'images')

# Exactly the 8-card pool this whole session's calibration work covers - Knight, Giant,
# Musketeer, MiniPekka, Minions, Archer, Fireball, Arrows. Nothing else has been validated,
# so nothing else belongs in this deck picker.
CARDS = ["Knight", "Giant", "Musketeer", "MiniPekka", "Minions", "Archer", "Fireball", "Arrows"]

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "static", "index.html"))


@app.get("/api/cards")
async def get_cards():
    out = []
    for name in CARDS:
        info = card_data.get(name, {})
        out.append({"name": name, "englishName": info.get("englishName", name),
                     "elixir": info.get("manaCost", 0)})
    return out


waiting_lobby = []  # list of WebSocket connections waiting for a pair
lobby_lock = asyncio.Lock()


import random

BOT_DECK = ["Knight", "Giant", "Musketeer", "MiniPekka", "Minions", "Archer", "Fireball", "Arrows"]


class Room:
    def __init__(self, ws0, ws1, bot=False):
        self.sockets = [ws0, ws1]
        self.bot = bot  # if True, player 1 is a simple scripted opponent, not a real socket
        self.decks = [None, BOT_DECK if bot else None]
        self.done_event = asyncio.Event()
        self.battle = None
        self.running = False

    async def send(self, idx, msg):
        if self.bot and idx == 1:
            return
        try:
            await self.sockets[idx].send_json(msg)
        except Exception:
            pass

    async def broadcast(self, msg):
        await asyncio.gather(self.send(0, msg), self.send(1, msg))

    def state_dict(self):
        return {
            "type": "state",
            "time": self.battle.time,
            "game_over": self.battle.game_over,
            "winner": self.battle.winner,
            "elixir": [p.elixir for p in self.battle.players],
            "hand": [p.cycle[:4] for p in self.battle.players],
            "tower_hp": [[p.king_tower_hp, p.left_tower_hp, p.right_tower_hp] for p in self.battle.players],
            "entities": [e.to_dict() for e in self.battle.entities.values() if e.is_alive],
        }

    async def run(self):
        for i, ws in enumerate(self.sockets):
            await self.send(i, {"type": "hello", "player_id": i})
        while not all(self.decks):
            for i, ws in enumerate(self.sockets):
                if self.decks[i] is not None or (self.bot and i == 1):
                    continue
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                    if msg.get("type") == "deck":
                        self.decks[i] = msg["cards"]
                except asyncio.TimeoutError:
                    continue
                except (WebSocketDisconnect, Exception):
                    return

        self.battle = battle.BattleState(
            player_mod.PlayerState(0, self.decks[0], 5.0),
            player_mod.PlayerState(1, self.decks[1], 5.0),
        )
        await self.broadcast({"type": "start"})
        self.running = True
        dt = 1 / 30  # 30Hz server tick is plenty smooth over a network and cheaper to host

        recv_tasks = [asyncio.create_task(self._bot_loop() if (self.bot and i == 1) else self._recv_loop(i))
                      for i in range(2)]
        try:
            while not self.battle.game_over:
                for _ in range(2):  # advance sim at 60 internal ticks/sec, broadcast at 30Hz
                    self.battle.step(1 / 60)
                await self.broadcast(self.state_dict())
                await asyncio.sleep(dt)
            await self.broadcast(self.state_dict())
        finally:
            for t in recv_tasks:
                t.cancel()

    async def _recv_loop(self, idx):
        ws = self.sockets[idx]
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "deploy" and self.battle:
                    ok = self.battle.deploy_card(idx, msg["card"], Position(msg["x"], msg["y"]))
                    # deploy_card fails silently (illegal tile, can't afford it, etc.) - without
                    # this, a rejected placement looks to the player exactly like a broken click
                    await self.send(idx, {"type": "deploy_result", "ok": ok, "card": msg["card"]})
        except (WebSocketDisconnect, Exception):
            return

    async def _bot_loop(self):
        """Not a trained AI (that's the next phase) - just enough scripted behavior to let
        one person test a full match alone: plays a random affordable card at a semi-sane
        spot on its own side every few seconds."""
        try:
            while not self.battle.game_over:
                await asyncio.sleep(random.uniform(2.0, 4.0))
                if self.battle.game_over:
                    break
                p1 = self.battle.players[1]
                playable = [c for c in p1.cycle[:4] if p1.can_play_card(c)]
                if not playable:
                    continue
                card = random.choice(playable)
                x = random.uniform(2, 16)
                y = random.uniform(18, 28)
                self.battle.deploy_card(1, card, Position(x, y))
        except Exception:
            return


pending_pairs = {}  # websocket -> asyncio.Future resolving to the Room it got paired into


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    partner = None
    async with lobby_lock:
        if waiting_lobby:
            partner = waiting_lobby.pop(0)
        else:
            waiting_lobby.append(websocket)
            pending_pairs[websocket] = asyncio.get_event_loop().create_future()

    if partner is None:
        await websocket.send_json({"type": "waiting"})
        try:
            room = await asyncio.wait_for(pending_pairs[websocket], timeout=5.0)
            await room.done_event.wait()
        except asyncio.TimeoutError:
            # nobody joined in time - fall back to a scripted bot opponent so a single
            # person can still test a full match without needing a second real player
            async with lobby_lock:
                if websocket in waiting_lobby:
                    waiting_lobby.remove(websocket)
                pending_pairs.pop(websocket, None)
            room = Room(websocket, None, bot=True)
            await room.run()
            room.done_event.set()
        except Exception:
            async with lobby_lock:
                if websocket in waiting_lobby:
                    waiting_lobby.remove(websocket)
                pending_pairs.pop(websocket, None)
            return
    else:
        room = Room(partner, websocket)
        fut = pending_pairs.pop(partner, None)
        if fut and not fut.done():
            fut.set_result(room)
        await room.run()
        room.done_event.set()
