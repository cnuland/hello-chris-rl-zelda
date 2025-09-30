"""Room ID mappings for The Legend of Zelda: Oracle of Seasons.

The overworld is a 16x16 grid (256 total screens, 0x00-0xFF).
Room IDs go left to right, top to bottom.

Based on Oracle of Seasons overworld map and gameplay testing.
"""

# Overworld room mappings (Holodrum)
# Note: The overworld is 16x16 = 256 screens total (0x00 to 0xFF)
# Each row has 16 screens (0x00-0x0F, 0x10-0x1F, 0x20-0x2F, etc.)

OVERWORLD_ROOMS = {
    # Northern Holodrum (Starting Area) - Rows 0xB0-0xC0
    0xB6: "Northern Holodrum - Starting Screen (Save State Location)",
    0xB5: "Northern Holodrum - West of Start",
    0xB7: "Northern Holodrum - East of Start",
    0xA6: "Northern Holodrum - North of Start (River)",
    0xC6: "Northern Holodrum - South of Start (Path to Village)",
    
    # Horon Village Area - Rows 0xC0-0xD0
    0xC5: "Horon Village - Northern Entrance",
    0xC6: "Horon Village - North Area",
    0xC7: "Horon Village - Northeast Area (near Maku Tree path)",
    0xD5: "Horon Village - West Side",
    0xD6: "Horon Village - Center (Town Square)",
    0xD7: "Horon Village - East Side (Maku Tree Path)",
    0xE5: "Horon Village - Southwest",
    0xE6: "Horon Village - South Center (Shop Area)",
    0xE7: "Horon Village - Southeast",
    
    # Maku Tree Area - East of Village
    0xD8: "Eastern Woods - Path to Maku Tree",
    0xD9: "Maku Tree - Entrance",
    0xC8: "Eastern Woods - North of Maku Tree",
    0xE8: "Eastern Woods - South of Maku Tree",
    
    # Hero's Cave (Wooden Sword Location) - Northeast of Village
    0xB8: "Hero's Cave - Exterior (Wooden Sword Location)",
    0xB9: "Eastern Holodrum - North",
    
    # Western Holodrum
    0xB0: "Western Holodrum - Far Northwest",
    0xB1: "Western Holodrum - North",
    0xC0: "Western Holodrum - West",
    0xC1: "Western Holodrum - Central",
    
    # Southern Holodrum
    0xF5: "Southern Holodrum - West",
    0xF6: "Southern Holodrum - Center",
    0xF7: "Southern Holodrum - East",
    
    # Known dungeon entrances and key locations
    # (These will be discovered during gameplay)
    0x28: "Gnarled Root Dungeon - Entrance (Dungeon 1)",
    0x44: "Snake's Remains - Entrance (Dungeon 2)",
    0x15: "Poison Moth's Lair - Entrance (Dungeon 3)",
    0x52: "Dancing Dragon Dungeon - Entrance (Dungeon 4)",
    0x79: "Unicorn's Cave - Entrance (Dungeon 5)",
    0x1A: "Ancient Ruins - Entrance (Dungeon 6)",
    0x6E: "Explorer's Crypt - Entrance (Dungeon 7)",
    0x0D: "Sword & Shield Maze - Entrance (Dungeon 8)",
    
    # Subrosia Portal locations (discovered during exploration)
    0x2B: "Subrosia Portal - North Woods",
    0x95: "Subrosia Portal - Sunken City",
    0x3C: "Subrosia Portal - Spool Swamp",
    0x78: "Subrosia Portal - Mt. Cucco",
    0xA4: "Subrosia Portal - Eyeglass Lake",
    
    # Temple of Seasons
    0x05: "Temple of Seasons - Entrance (North Woods)",
    
    # Other key locations
    0x9B: "Sunken City - North",
    0xAB: "Sunken City - Center",
    0xBB: "Sunken City - South",
    0x2A: "North Horon - Woods Area",
    0x3A: "Spool Swamp - Entrance",
    0x4A: "Spool Swamp - Central",
    0x78: "Mt. Cucco - Base",
    0x68: "Mt. Cucco - Summit",
    0xA4: "Eyeglass Lake - Shore",
    0xA5: "Eyeglass Lake - East Side",
}

# Subrosia room mappings (separate map accessed via portals)
SUBROSIA_ROOMS = {
    # Subrosia has its own map layout
    # These are accessed when level_bank indicates Subrosia
    0x00: "Subrosia - Market",
    0x10: "Subrosia - House of Pirates",
    0x20: "Subrosia - Dance Hall",
    0x30: "Subrosia - Furnace",
    0x40: "Subrosia - Smithy",
}

# Region classifications for strategic guidance
REGION_CLASSIFICATIONS = {
    "northern_holodrum": list(range(0xA0, 0xB0)) + list(range(0xB0, 0xC0)),
    "horon_village": list(range(0xC5, 0xC8)) + list(range(0xD5, 0xD8)) + list(range(0xE5, 0xE8)),
    "eastern_woods": [0xC8, 0xC9, 0xD8, 0xD9, 0xE8, 0xE9],
    "western_holodrum": list(range(0xB0, 0xB5)) + list(range(0xC0, 0xC5)),
    "southern_holodrum": list(range(0xF0, 0x100)),
}


def get_room_name(room_id: int, level_bank: int = 0) -> str:
    """Get human-readable room name from room ID.
    
    Args:
        room_id: Room ID from memory (0x00-0xFF for overworld)
        level_bank: Current level bank (0 = overworld, other values = dungeons/Subrosia)
    
    Returns:
        Human-readable room name or "Unknown Room"
    """
    if level_bank == 0:
        # Overworld/Holodrum
        return OVERWORLD_ROOMS.get(room_id, f"Holodrum Area {room_id:02X}")
    elif level_bank == 4:
        # Subrosia
        return SUBROSIA_ROOMS.get(room_id, f"Subrosia Area {room_id:02X}")
    else:
        # Dungeon
        return f"Dungeon {level_bank} - Room {room_id:02X}"


def get_region_from_room(room_id: int) -> str:
    """Get region classification from room ID.
    
    Args:
        room_id: Room ID from memory
    
    Returns:
        Region name for strategic context
    """
    for region, room_ids in REGION_CLASSIFICATIONS.items():
        if room_id in room_ids:
            return region.replace("_", " ").title()
    
    return "Unknown Region"


def is_in_horon_village(room_id: int) -> bool:
    """Check if room ID is within Horon Village."""
    return room_id in REGION_CLASSIFICATIONS["horon_village"]


def is_near_maku_tree(room_id: int) -> bool:
    """Check if room ID is near the Maku Tree."""
    maku_tree_rooms = [0xC7, 0xD7, 0xD8, 0xD9, 0xE7, 0xE8]
    return room_id in maku_tree_rooms


def get_strategic_context(room_id: int, has_sword: bool = False) -> str:
    """Get strategic context based on current room and game progression.
    
    Args:
        room_id: Current room ID
        has_sword: Whether Link has acquired the wooden sword
    
    Returns:
        Strategic guidance text
    """
    if not has_sword:
        if is_near_maku_tree(room_id):
            return "🌳 Near Maku Tree! Look for the entrance to get your sword."
        elif is_in_horon_village(room_id):
            return "🏘️ In Horon Village. Head EAST to find the Maku Tree."
        elif room_id in range(0xB0, 0xC0):
            return "🌲 Northern Holodrum. Head SOUTH to reach Horon Village."
    
    # Post-sword guidance
    dungeon_rooms = {
        0x28: "Gnarled Root Dungeon (D1) nearby - first dungeon!",
        0x44: "Snake's Remains (D2) nearby - bring bombs",
        0x15: "Poison Moth's Lair (D3) nearby",
        0x52: "Dancing Dragon Dungeon (D4) nearby",
        0x79: "Unicorn's Cave (D5) nearby",
        0x1A: "Ancient Ruins (D6) nearby",
        0x6E: "Explorer's Crypt (D7) nearby",
        0x0D: "Sword & Shield Maze (D8) nearby - final dungeon!",
    }
    
    if room_id in dungeon_rooms:
        return f"⚔️ {dungeon_rooms[room_id]}"
    
    return ""
