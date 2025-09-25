"""Memory addresses for The Legend of Zelda: Oracle of Seasons.

Based on community disassembly and RAM mapping projects.
Addresses may vary by ROM version - these are for US v1.0.
"""

# Player state addresses
PLAYER_X = 0xD300  # Link's X position
PLAYER_Y = 0xD301  # Link's Y position
PLAYER_DIRECTION = 0xD302  # Facing direction (0-3)
PLAYER_ROOM = 0xD303  # Current room/screen ID
PLAYER_HEALTH = 0xD304  # Current health (hearts)
PLAYER_MAX_HEALTH = 0xD305  # Maximum health

# Inventory addresses
RUPEES = 0xD310  # Current rupees (2 bytes, little endian)
KEYS = 0xD312  # Small keys count
SWORD_LEVEL = 0xD313  # Sword upgrade level
SHIELD_LEVEL = 0xD314  # Shield upgrade level

# Items (bit flags)
INVENTORY_1 = 0xD320  # First inventory byte
INVENTORY_2 = 0xD321  # Second inventory byte
INVENTORY_3 = 0xD322  # Third inventory byte

# Dungeon progress
DUNGEON_KEYS = 0xD330  # Dungeon-specific small keys
BOSS_KEYS = 0xD331  # Boss key flags
DUNGEON_MAP = 0xD332  # Dungeon map obtained
DUNGEON_COMPASS = 0xD333  # Compass obtained

# Season state
CURRENT_SEASON = 0xD340  # Current season (0-3: Spring, Summer, Autumn, Winter)
SEASON_SPIRITS = 0xD341  # Season spirit locations found

# Game flags
GAME_FLAGS_START = 0xD400  # Start of game event flags
GAME_FLAGS_END = 0xD4FF   # End of game event flags

# Enemy/interaction state
ENEMIES_DEFEATED = 0xD500  # Enemies defeated in current room
CHESTS_OPENED = 0xD501    # Chests opened flags
SWITCHES_ACTIVATED = 0xD502  # Switch states

# Screen transition
SCREEN_TRANSITION = 0xD600  # Screen transition state
LOADING_SCREEN = 0xD601    # Loading/transition in progress

# Menu state
MENU_STATE = 0xD700  # Current menu state
ITEM_SELECTED = 0xD701  # Selected item in menu

# Specific item flags (within inventory bytes)
ITEM_FLAGS = {
    'gale_boomerang': (INVENTORY_1, 0x01),
    'shovel': (INVENTORY_1, 0x02),
    'bracelet': (INVENTORY_1, 0x04),
    'feather': (INVENTORY_1, 0x08),
    'seed_satchel': (INVENTORY_1, 0x10),
    'flippers': (INVENTORY_1, 0x20),
    'magnet_gloves': (INVENTORY_1, 0x40),
    'rod_of_seasons': (INVENTORY_1, 0x80),

    'slingshot': (INVENTORY_2, 0x01),
    'boomerang': (INVENTORY_2, 0x02),
    'rod_of_somaria': (INVENTORY_2, 0x04),
    'magnetic_gloves': (INVENTORY_2, 0x08),
    'spring_banana': (INVENTORY_2, 0x10),
    'ricky_gloves': (INVENTORY_2, 0x20),
    'dimitri_flippers': (INVENTORY_2, 0x40),
    'moosh_flute': (INVENTORY_2, 0x80),
}

# Boss defeated flags
BOSS_FLAGS = {
    'gohma': (BOSS_KEYS, 0x01),
    'dodongo': (BOSS_KEYS, 0x02),
    'aquamentus': (BOSS_KEYS, 0x04),
    'gleeok': (BOSS_KEYS, 0x08),
    'digdogger': (BOSS_KEYS, 0x10),
    'manhandla': (BOSS_KEYS, 0x20),
    'patra': (BOSS_KEYS, 0x40),
    'onox': (BOSS_KEYS, 0x80),
}

# Season constants
SEASONS = {
    0: 'spring',
    1: 'summer',
    2: 'autumn',
    3: 'winter'
}

# Direction constants
DIRECTIONS = {
    0: 'down',
    1: 'up',
    2: 'left',
    3: 'right'
}