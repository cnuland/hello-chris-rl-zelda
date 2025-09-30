"""Memory addresses for The Legend of Zelda: Oracle of Seasons.

Based on Data Crystal community RAM mapping and oracles-disasm project.
Addresses are for US v1.0 ROM. Source: https://datacrystal.romhacking.net/
"""

# Core player position (Data Crystal confirmed)
PLAYER_X = 0xC4AC  # Link's X position (pixel coordinate within screen)
PLAYER_Y = 0xC4AD  # Link's Y position (pixel coordinate within screen)
PLAYER_DIRECTION = 0xD302  # Facing direction (needs verification)
PLAYER_ROOM = 0xC63B  # Current overworld room/screen ID (0x00-0xFF)
CURRENT_LEVEL_BANK = 0xC63A  # Current level bank (overworld vs dungeon)
CURRENT_DUNGEON_POSITION = 0xC63C  # Current dungeon room position
CURRENT_DUNGEON_FLOOR = 0xC63D  # Current dungeon floor number

# Health system (Data Crystal confirmed)
PLAYER_HEALTH = 0xC6A2  # Current hearts (quarter-hearts, divide by 4)
PLAYER_MAX_HEALTH = 0xC6A3  # Maximum hearts (quarter-hearts, divide by 4)
HEART_PIECES = 0xC6A4  # Heart pieces (0-3, becomes heart at 4)

# Resources (Data Crystal confirmed)
RUPEES = 0xC6A5  # Current rupees (2 bytes, decimal)
ORE_CHUNKS = 0xC6A7  # Ore chunks (2 bytes)
SHIELD_LEVEL = 0xC6A9  # Shield upgrade level
CURRENT_BOMBS = 0xC6AA  # Current bomb count
MAX_BOMBS = 0xC6AB  # Maximum bomb capacity
SWORD_LEVEL = 0xC6AC  # Sword upgrade level
CURRENT_BOMBCHUS = 0xC6AD  # Current Bombchu count
SEED_SATCHEL_LEVEL = 0xC6AE  # Seed carrying capacity level

# Active items (Data Crystal confirmed)
B_BUTTON_ITEM = 0xC680  # Item assigned to B button
A_BUTTON_ITEM = 0xC681  # Item assigned to A button

# Inventory storage (Data Crystal confirmed)
INVENTORY_START = 0xC682  # Start of 16-byte inventory storage
INVENTORY_SIZE = 16  # 16 bytes of inventory data

# Ring system (Data Crystal confirmed) 
VASU_RING_FLAGS = 0xC6CA  # Ring flags and special ring conditions
RING_BOX_LEVEL = 0xC6C6  # Ring box capacity level
RINGS_OWNED_START = 0xC616  # Start of ring ownership flags (8 bytes)
RINGS_BUFFER_START = 0xC5C0  # Ring buffer/flags (64 bytes)

# Seed counts (Data Crystal confirmed)
EMBER_SEEDS = 0xC6B5  # Ember Seeds count
SCENT_SEEDS = 0xC6B6  # Scent Seeds count  
PEGASUS_SEEDS = 0xC6B7  # Pegasus Seeds count
GALE_SEEDS = 0xC6B8  # Gale Seeds count
MYSTERY_SEEDS = 0xC6B9  # Mystery Seeds count
GASHA_SEEDS = 0xC6BA  # Gasha Seeds count

# Equipment levels (Data Crystal confirmed)
FLUTE_TYPE = 0xC6AF  # Which flute/companion (Dimitri, Ricky, Moosh)
SEASONS_OBTAINED = 0xC6B0  # Which seasons unlocked
BOOMERANG_LEVEL = 0xC6B1  # Boomerang upgrade level
MAGNETIC_GLOVES = 0xC6B2  # Magnetic gloves polarity/flags
SLINGSHOT_LEVEL = 0xC6B3  # Slingshot upgrade level
ROCS_FEATHER_LEVEL = 0xC6B4  # Roc's Feather jump upgrade

# Progress tracking (Data Crystal confirmed)
ESSENCES_COLLECTED = 0xC6BB  # Essences of Nature bitmask
PLAYER_NAME = 0xC602  # Player name (6 bytes, null-terminated)
TOTAL_DEATHS = 0xC61E  # Total death count (2 bytes)
ENEMIES_KILLED = 0xC620  # Cumulative enemies killed (2 bytes)
TIME_PASSED = 0xC622  # Time since file creation (4 bytes)
RUPEES_COLLECTED = 0xC627  # Cumulative rupees collected (2 bytes)

# World state (Data Crystal confirmed)
CURRENT_LEVEL_BANK = 0xC63A  # Current map bank
OVERWORLD_POSITION = 0xC63B  # Absolute overworld position (0x00-0xFF)
DUNGEON_POSITION = 0xC63C  # Current dungeon room position  
DUNGEON_FLOOR = 0xC63D  # Current dungeon floor number
MAPLE_COUNTER = 0xC63E  # Maple encounter counter

# Screen/level data (Data Crystal confirmed)
OVERWORLD_FLAGS_START = 0xC700  # Screen flags (256 bytes): chests, events, etc.
ENEMIES_ON_SCREEN = 0xCC30  # Count of active enemy sprites
COLLISION_MAP_START = 0xCE00  # Current screen collision data (~0xB0 bytes)
TILE_DATA_START = 0xCF00  # Current screen tile indices (~0xB0 bytes)

# Settings (Data Crystal confirmed)
SOUND_VOLUME = 0xC024  # Sound volume control bits
TEXT_SPEED = 0xC629  # Message/text display speed (0x00-0x04)
CUTSCENE_INDEX = 0xC2EF  # Current cutscene index

# Gasha seed planting (Data Crystal confirmed - specific locations)
GASHA_SEEDS_PLANTED = {
    'B-16': 0xC64C,
    'C-3': 0xC64D, 
    'D-9': 0xC64E,
    'D-12': 0xC64F,
    'E-5': 0xC650,
    'D-16': 0xC651,
    'H-6': 0xC652,
    'I-1': 0xC653
}

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