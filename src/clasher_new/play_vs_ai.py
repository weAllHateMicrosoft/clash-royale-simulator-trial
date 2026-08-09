"""
Play locally against the pretrained PPO agent (no networking needed).
You are Player 0 (blue, bottom of screen).

Flow:
  - Pick up to 8 cards from the full card pool, click Confirm.
  - Click a hand card, then click the arena to deploy it.
  - After the match ends, press R to pick a new deck and play again, or Esc to quit.

Controls (in-match):
  - Click a card in your hand (bottom bar), then click a legal tile in the arena to deploy.
  - SPACE: pause/unpause
  - 1-5: simulation speed multiplier
  - ESC: quit,  R (after game over): rematch / new deck

Usage:
  python play_vs_ai.py [checkpoint_name]
  (checkpoint_name defaults to cr_checkpoint, the other option in this repo is cr_discrete)

Note: the AI only knows the 8 cards it was trained on (Knight, MiniPekka, Arrows, Minions,
Musketeer, Fireball, Giant, Archer) - that's baked into its network (fixed-size card embedding
table), so its own deck always stays that one. You can deploy anything from the full pool;
cards it never saw during training just won't be recognized by its "identity" sense (they still
carry real hp/damage/speed/range stats and it'll react to those - just don't expect it to know
say a Prince from a Giant Skeleton).
"""
import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO

import battle, player
from core import Position
from card_utils import Card, card_data
from environment import entity_names, card_types, player_1_deck

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "cr_checkpoint"

pygame.init()
TILE = 22
AX, AY = 50, 50
AW, AH = 18 * TILE, 32 * TILE
HAND_H = 110
W, H = AW + 120, AH + 100 + HAND_H
BLUE, RED, GREEN, CYAN, DKGRAY, BLACK, WHITE, YELLOW, GREY = (
    (100, 100, 255), (255, 100, 100), (100, 255, 100), (100, 255, 255),
    (64, 64, 64), (0, 0, 0), (255, 255, 255), (230, 200, 40), (150, 150, 150)
)

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"Clash Royale Simulator - You vs {CHECKPOINT}")
font = pygame.font.Font(None, 20)
small_font = pygame.font.Font(None, 16)
big_font = pygame.font.Font(None, 34)
clock = pygame.time.Clock()

CARDS = [
    "Knight", "Giant", "Archer", "Goblins", "Pekka", "MiniPekka",
    "Minions", "Skeletons", "SkeletonArmy", "Balloon", "Witch",
    "Barbarians", "Golem", "Valkyrie", "Bomber", "Musketeer",
    "BabyDragon", "Prince", "Wizard", "SpearGoblins",
    "GiantSkeleton", "HogRider", "MinionHorde", "RoyalGiant",
    "Princess", "ThreeMusketeers", "BlowdartGoblin", "AngryBarbarians",
    "Bats", "DartBarrell", "RoyalHogs", "Cannon", "Xbow",
    "IceWizard", "SkeletonWarriors", "DarkPrince", "LavaHound",
    "IceSpirits", "FireSpirits", "Miner", "Bowler",
    "Rage", "RageBarbarian", "BattleRam", "Fireball", "Arrows"
]
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "client_side", "images")
_img_cache = {}


def load_card_image(name, size):
    """For the deck builder: always returns something (grey box if art is missing)."""
    key = (name, size)
    if key in _img_cache:
        return _img_cache[key]
    english = card_data[name].get("englishName", name)
    path = os.path.join(IMAGES_DIR, f"{english}.png")
    try:
        img = pygame.transform.scale(pygame.image.load(path), size)
    except Exception:
        img = pygame.Surface(size)
        img.fill((80, 80, 80))
    _img_cache[key] = img
    return img


def try_load_card_image(name, size):
    """For arena tokens: real Supercell icon or nothing - never a fabricated placeholder."""
    if name not in card_data:
        return None
    key = ("token", name, size)
    if key in _img_cache:
        return _img_cache[key]
    english = card_data[name].get("englishName", name)
    path = os.path.join(IMAGES_DIR, f"{english}.png")
    if not os.path.exists(path):
        _img_cache[key] = None
        return None
    img = pygame.transform.scale(pygame.image.load(path).convert_alpha(), size)
    _img_cache[key] = img
    return img


def safe_entity_id(name):
    """entity_names only covers the AI's trained 8-card deck + towers/spells.
    Anything else (cards the human picks from the full pool) falls back to
    the 'None' slot so it doesn't crash the embedding lookup - real stats
    (hp/damage/speed/range/etc.) still flow through the rest of the obs vector."""
    return entity_names.index(name) if name in entity_names else 0


def w2s(x, y):
    y = 32 - y
    return int(AX + x * TILE), int(AY + y * TILE)


def s2w(sx, sy):
    return (sx - AX) / TILE, 32 - (sy - AY) / TILE


def observe(b: "battle.BattleState", player_id_observe):
    """Same construction as environment.CREnv.observe, decoupled from the gym env
    so we don't touch its module-level deck lists (CREnv.reset shuffles them in place)."""
    obs = np.zeros((32, 18, 15), dtype=np.float32)
    for eid, each in b.entities.items():
        if not each.is_alive:
            continue
        if isinstance(each, battle.Projectile):
            continue
        entity_id = safe_entity_id(each.name)
        card_type = card_types.index(each.data.type) if each.data.type in card_types else card_types.index('character')
        player_id = each.player
        elixir = each.data.elixir
        is_air = int(each.data.is_air_unit)
        attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)
        speed = each.data.speed
        hp_left = np.log(max(each.hp, 1)) / 10
        hp_percentage = each.hp / each.data.hp if each.data.hp != 0 else 0
        hit_speed = each.data.hit_speed
        attack_range = each.data.range / 3
        sight_range = each.data.sight_range / 3
        damage = each.data.damage / 200
        projectile_damage = each.data.projectile_data.damage / 200

        x, y = int(each.position.x), int(each.position.y)
        if player_id == 1:
            x = 17 - x
            y = 31 - y
        obs_arr = np.array([entity_id, player_id, elixir, card_type, speed, is_air, attacks_ground, attacks_air,
                             hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
        if 0 <= y < 32 and 0 <= x < 18:
            obs[y][x] = obs_arr.copy()

    hand = np.array([safe_entity_id(each) for each in b.players[player_id_observe].cycle[:5]], dtype=np.int32)
    return {
        'grid': obs,
        'hand': hand,
        'elixir': np.array([b.players[player_id_observe].elixir], dtype=np.float32),
    }


def ai_take_action(b: "battle.BattleState", model: PPO):
    obs1 = observe(b, 1)
    action, _ = model.predict(obs1, deterministic=True)
    slot, y, x = action
    p1 = b.players[1]
    if slot != 0:
        card_name = p1.cycle[slot - 1]
        b.deploy_card(1, card_name, Position(18 - (x + 0.5), 32 - (y + 0.5)))


def card_selection_screen():
    """Pick up to 8 cards for your deck. Returns a list of internal card names."""
    selected = []
    cols = 7
    cw, ch = 70, 100
    gap = 8
    running = True
    while running:
        screen.fill((25, 25, 30))
        title = big_font.render(f"Build your deck ({len(selected)}/8)", True, WHITE)
        screen.blit(title, (20, 15))
        rects = []
        for i, name in enumerate(CARDS):
            col, row = i % cols, i // cols
            x, y = 20 + col * (cw + gap), 60 + row * (ch + 22)
            rect = pygame.Rect(x, y, cw, ch)
            rects.append((name, rect))
            img = load_card_image(name, (cw, ch))
            screen.blit(img, rect)
            if name in selected:
                pygame.draw.rect(screen, (60, 220, 60), rect, 3)
            lbl = small_font.render(name[:11], True, (210, 210, 210))
            screen.blit(lbl, (x, y + ch + 2))

        confirm_rect = pygame.Rect(W - 170, H - 60, 140, 40)
        if len(selected) == 8:
            pygame.draw.rect(screen, (0, 170, 0), confirm_rect, border_radius=6)
            screen.blit(font.render("Confirm", True, WHITE), font.render("Confirm", True, WHITE).get_rect(center=confirm_rect.center))
        clear_rect = pygame.Rect(20, H - 60, 140, 40)
        pygame.draw.rect(screen, (120, 40, 40), clear_rect, border_radius=6)
        screen.blit(font.render("Clear", True, WHITE), font.render("Clear", True, WHITE).get_rect(center=clear_rect.center))

        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if len(selected) == 8 and confirm_rect.collidepoint(ev.pos):
                    return selected
                if clear_rect.collidepoint(ev.pos):
                    selected = []
                    continue
                for name, rect in rects:
                    if rect.collidepoint(ev.pos):
                        if name in selected:
                            selected.remove(name)
                        elif len(selected) < 8:
                            selected.append(name)
        clock.tick(30)


def draw_arena():
    pygame.draw.rect(screen, GREEN, (AX, AY, AW, AH))
    ry = AY + 15 * TILE
    pygame.draw.rect(screen, CYAN, (AX, ry, AW, 2 * TILE))
    for bx in [2, 13]:
        pygame.draw.rect(screen, DKGRAY, (AX + bx * TILE, ry, 3 * TILE, 2 * TILE))
    for x in range(19):
        pygame.draw.line(screen, (0, 150, 0), (AX + x * TILE, AY), (AX + x * TILE, AY + AH), 1)
    for y in range(33):
        pygame.draw.line(screen, (0, 150, 0), (AX, AY + y * TILE), (AX + AW, AY + y * TILE), 1)


def draw_entities(b: "battle.BattleState"):
    for e in b.entities.values():
        if not e.is_alive:
            continue
        sx, sy = w2s(e.position.x, e.position.y)
        color = BLUE if e.player == 0 else RED
        r = int(e.data.collision_radius * TILE)
        if isinstance(e, battle.Projectile):
            pygame.draw.circle(screen, color, (sx, sy), max(r, 4), 2)
        else:
            icon_size = max(r * 2, 22)
            icon = try_load_card_image(e.name, (icon_size, icon_size))
            if icon:
                screen.blit(icon, icon.get_rect(center=(sx, sy)))
                pygame.draw.circle(screen, color, (sx, sy), icon_size // 2, 3)
            else:
                pygame.draw.circle(screen, color, (sx, sy), max(r, 4))
        pygame.draw.circle(screen, BLACK, (sx, sy), max(r, 4), 1)
        lbl = small_font.render(e.name, True, BLACK)
        screen.blit(lbl, lbl.get_rect(center=(sx, sy + r + 10)))
        if e.hp > 0:
            bw = max(r * 2, 16)
            pygame.draw.rect(screen, BLACK, (sx - bw // 2 - 1, sy - r - 12, bw + 2, 5))
            hp_width = (e.hp / e.data.hp) * bw if not e.shield_health else (e.shield_health / e.data.shield_health) * bw
            pygame.draw.rect(screen, GREEN, (sx - bw // 2, sy - r - 11, hp_width, 3))
            if isinstance(e, battle.Building):
                hp_txt = small_font.render(str(int(e.hp)), True, WHITE)
                screen.blit(hp_txt, hp_txt.get_rect(center=(sx, sy)))


def draw_hand(b: "battle.BattleState", selected_slot):
    p0 = b.players[0]
    hand = p0.cycle[:4]
    hand_y = AY + AH + 40
    card_w = (AW - 3 * 8) // 4
    for i, name in enumerate(hand):
        x = AX + i * (card_w + 8)
        rect = pygame.Rect(x, hand_y, card_w, 70)
        cost = Card(name).elixir
        can_play = p0.elixir >= cost
        color = (60, 90, 160) if can_play else (70, 70, 70)
        if selected_slot == i:
            color = YELLOW
        pygame.draw.rect(screen, color, rect, border_radius=6)
        pygame.draw.rect(screen, BLACK, rect, 2, border_radius=6)
        lbl = small_font.render(name, True, WHITE if can_play or selected_slot == i else GREY)
        screen.blit(lbl, lbl.get_rect(center=(rect.centerx, rect.centery - 10)))
        cost_lbl = font.render(str(cost), True, WHITE if can_play or selected_slot == i else GREY)
        screen.blit(cost_lbl, cost_lbl.get_rect(center=(rect.centerx, rect.centery + 14)))


def draw_elixir_bar(b: "battle.BattleState"):
    bar_y = AY + AH + 15
    elixir = b.players[0].elixir
    pygame.draw.rect(screen, (40, 40, 40), (AX, bar_y, AW, 14))
    pygame.draw.rect(screen, (160, 32, 240), (AX, bar_y, int((elixir / 10.0) * AW), 14))
    txt = small_font.render(f"{elixir:.1f}/10", True, WHITE)
    screen.blit(txt, txt.get_rect(center=(AX + AW // 2, bar_y + 7)))


def draw_ui(b: "battle.BattleState", paused, speed):
    text = f"t={b.time:.1f}s  speed={speed}x"
    if b.game_over:
        text += f"   {'YOU WIN' if b.winner == 0 else 'AI WINS'}  -  press R for a new deck, Esc to quit"
    if paused:
        text += "   [PAUSED]"
    txt = font.render(text, True, BLACK)
    screen.blit(txt, (AX, AY + AH + HAND_H + 55))


def play_match(model, human_deck):
    b = battle.BattleState(
        player.PlayerState(0, human_deck[:], 5.0),
        player.PlayerState(1, player_1_deck[:], 5.0),
    )

    paused = False
    speed = 1
    selected_slot = None
    ai_decision_interval = 0.5
    next_ai_decision = 0.0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif pygame.K_1 <= ev.key <= pygame.K_5:
                    speed = ev.key - pygame.K_0
                elif ev.key == pygame.K_r and b.game_over:
                    return
            elif ev.type == pygame.MOUSEBUTTONDOWN and not b.game_over:
                mx, my = ev.pos
                hand_y = AY + AH + 40
                card_w = (AW - 3 * 8) // 4
                if hand_y <= my <= hand_y + 70:
                    idx = (mx - AX) // (card_w + 8)
                    if 0 <= idx < 4 and AX <= mx <= AX + 4 * (card_w + 8):
                        selected_slot = idx if selected_slot != idx else None
                elif AY <= my <= AY + AH and selected_slot is not None:
                    wx, wy = s2w(mx, my)
                    card_name = b.players[0].cycle[selected_slot]
                    if b.deploy_card(0, card_name, Position(wx, wy)):
                        selected_slot = None

        if not paused and not b.game_over:
            for _ in range(speed):
                b.step(1 / 60)
                if b.time >= next_ai_decision:
                    ai_take_action(b, model)
                    next_ai_decision = b.time + ai_decision_interval

        screen.fill(WHITE)
        draw_arena()
        draw_entities(b)
        draw_elixir_bar(b)
        draw_hand(b, selected_slot)
        draw_ui(b, paused, speed)
        pygame.display.flip()
        clock.tick(60)


def main():
    print(f"Loading model {CHECKPOINT}.zip ...")
    model = PPO.load(CHECKPOINT, device="cpu")
    while True:
        deck = card_selection_screen()
        play_match(model, deck)


if __name__ == "__main__":
    main()
