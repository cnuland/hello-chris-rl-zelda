# 🔍 Oracle of Seasons Entity Detection Research

**Date**: October 16, 2025  
**Sources**: ZeldaHacking.net Wiki, Data Crystal

---

## 📊 Entity Memory Architecture

Oracle of Seasons uses a **structured entity system** with dedicated memory regions for different entity types. Each entity occupies **64 bytes** of contiguous memory.

### Entity Categories and Memory Ranges

| Category | Address Range | Slots | Description |
|----------|---------------|-------|-------------|
| **Special Objects** | `0xD000 - 0xD03F` | 1 | Link (slot 0), animal companion, Maple, raft |
| **Parent Items** | `0xD040 - 0xD07F` | 1 | State-holders for active items (not rendered) |
| **Items** | `0xD080 - 0xD0FF` | 2 | Weapons, projectiles (swords, arrows, etc.) |
| **Interactions** | `0xD140 - 0xD17F` | 1 | **NPCs**, scripts, non-hostile entities |
| **Enemies** | `0xD180 - 0xD1BF` | 1 | **Hostile entities** that can damage Link |
| **Parts** | `0xD1C0 - 0xD1FF` | 1 | Enemy weapons, projectiles, owl statues |

**Source**: [wiki.zeldahacking.net/oracle/Objects](https://wiki.zeldahacking.net/oracle/Objects)

---

## 🎯 Key Findings

### Entity Slots

Each 64-byte entity slot contains:
- **Type** (identifies what kind of entity)
- **Position** (X, Y coordinates)
- **Health/State** (HP, active/inactive)
- **Direction** (facing direction)
- **Animation frame** (current sprite)
- **Behavior state** (AI state machine)

### Multiple Entities Per Category

- **Items**: `0xD080 - 0xD0FF` = 128 bytes = **2 slots** (64 bytes each)
- **Interactions (NPCs)**: `0xD140 - 0xD17F` = 64 bytes = **1 slot**
- **Enemies**: `0xD180 - 0xD1BF` = 64 bytes = **1 slot**
- **Parts**: `0xD1C0 - 0xD1FF` = 64 bytes = **1 slot**

**Note**: The address ranges suggest **limited simultaneous entities** per category, which is typical for Game Boy Color memory constraints.

---

## 🔬 Proposed Entity Structure (64 bytes per entity)

Based on similar Game Boy Zelda games (NES Zelda, Link's Awakening), the 64-byte structure likely contains:

```
Offset | Bytes | Field              | Description
-------|-------|--------------------|-----------------------------------------
+0x00  | 1     | Entity Type        | ID of entity (0x00 = inactive)
+0x01  | 1     | Entity Status      | Active/inactive, flags
+0x02  | 1     | X Position (High)  | Screen X coordinate (high byte)
+0x03  | 1     | X Position (Low)   | Screen X coordinate (low byte)
+0x04  | 1     | Y Position (High)  | Screen Y coordinate (high byte)
+0x05  | 1     | Y Position (Low)   | Screen Y coordinate (low byte)
+0x06  | 1     | Z Position         | Vertical offset (flying/jumping)
+0x07  | 1     | Direction          | Facing direction (0-3 or 0-7)
+0x08  | 1     | Animation Frame    | Current sprite frame
+0x09  | 1     | Animation Counter  | Frame counter for animation
+0x0A  | 1     | Health             | Current HP (enemies only)
+0x0B  | 1     | Max Health         | Maximum HP
+0x0C  | 1     | Behavior State     | AI state (idle, chase, attack, etc.)
+0x0D  | 1     | Timer              | Generic timer for behavior
+0x0E  | 1     | Speed              | Movement speed
+0x0F  | 1     | Flags              | Various entity flags
+0x10+ | 48    | Entity-specific    | Additional data per entity type
```

**Note**: This is an **educated guess** based on NES Zelda and other Game Boy titles. Actual structure requires runtime memory inspection.

---

## 🎮 Already Confirmed Addresses

From our existing `zelda_addresses.py`:

```python
# Confirmed working:
ENEMIES_ON_SCREEN = 0xCC30  # Count of active enemy sprites ✅

# Player position (confirmed working in logs):
PLAYER_X = 0xC4AC  # Link's X position (varies correctly: 1→3) ✅
PLAYER_Y = 0xC4AD  # Link's Y position (STUCK AT 0 - needs fix) ⚠️
```

---

## 🚀 Implementation Strategy

### Phase 1: Count Active Entities (Simple)

Read **entity type field** (`offset +0x00`) for each slot to count active entities:

```python
# Interactions (NPCs)
NPC_BASE = 0xD140
npc_count = 0
if pyboy_bridge.get_memory(NPC_BASE + 0x00) != 0x00:  # Type != 0 means active
    npc_count += 1

# Enemies
ENEMY_BASE = 0xD180
enemy_count = 0
if pyboy_bridge.get_memory(ENEMY_BASE + 0x00) != 0x00:
    enemy_count += 1

# Items/Projectiles
ITEM_BASE_1 = 0xD080
ITEM_BASE_2 = 0xD0C0  # Second item slot
item_count = 0
if pyboy_bridge.get_memory(ITEM_BASE_1 + 0x00) != 0x00:
    item_count += 1
if pyboy_bridge.get_memory(ITEM_BASE_2 + 0x00) != 0x00:
    item_count += 1
```

### Phase 2: Detailed Entity Data (Advanced)

If simple counting works, read full entity data:

```python
def read_entity(pyboy_bridge, base_address):
    """Read full entity data from 64-byte slot."""
    return {
        'type': pyboy_bridge.get_memory(base_address + 0x00),
        'status': pyboy_bridge.get_memory(base_address + 0x01),
        'x': pyboy_bridge.get_memory(base_address + 0x02) << 8 | 
             pyboy_bridge.get_memory(base_address + 0x03),
        'y': pyboy_bridge.get_memory(base_address + 0x04) << 8 | 
             pyboy_bridge.get_memory(base_address + 0x05),
        'direction': pyboy_bridge.get_memory(base_address + 0x07),
        'health': pyboy_bridge.get_memory(base_address + 0x0A),
    }
```

---

## ⚠️ Limitations and Unknowns

### Known Limitations

1. **Single Entity Slots**:
   - Only 1 NPC can be active at a time (64 bytes)
   - Only 1 enemy can be active at a time (64 bytes)
   - This seems **incorrect** for gameplay (multiple enemies exist)

2. **Possible Solutions**:
   - Multiple slots may exist (need to scan wider ranges)
   - Entities may be stored in **arrays** (e.g., 8 enemies × 64 bytes = 512 bytes)
   - OR the ranges are **per-type** (e.g., 0xD180-0xD1BF is ALL enemies, not just 1)

3. **Entity Count Discrepancy**:
   - If `0xD180-0xD1BF` (64 bytes) is for ALL enemies, then it's an **entity table**, not individual slots
   - Need to verify: Is this a **single entity** or an **array of entities**?

### What We Don't Know

1. **Exact entity type field offset** (guessed as +0x00)
2. **Exact position field offsets** (guessed as +0x02/+0x03 for X, +0x04/+0x05 for Y)
3. **How many entities can exist simultaneously**
4. **Whether these are entity arrays or single slots**

---

## 🧪 Testing Strategy

### Test 1: Simple Type Check (Safest)

```python
# In a known area with NPCs (e.g., Horon Village):
npc_type = pyboy_bridge.get_memory(0xD140)  # Should be non-zero if NPC present
enemy_type = pyboy_bridge.get_memory(0xD180)  # Should be non-zero if enemy present

print(f"NPC type: {npc_type:02X}, Enemy type: {enemy_type:02X}")
```

Expected:
- Horon Village: NPC type > 0x00, enemy type = 0x00
- Dungeon: NPC type = 0x00, enemy type > 0x00
- Overworld: Both may be active

### Test 2: Scan for Active Entities

```python
# Scan wider ranges to find entity arrays:
for i in range(0xD140, 0xD200, 64):  # Check every 64 bytes
    entity_type = pyboy_bridge.get_memory(i)
    if entity_type != 0x00:
        print(f"Active entity at 0x{i:04X}: type={entity_type:02X}")
```

### Test 3: Use Existing Count Address

```python
# We already have this confirmed address:
enemy_count = pyboy_bridge.get_memory(0xCC30)
print(f"Enemies on screen (from 0xCC30): {enemy_count}")
```

This is the **safest** approach - we know `0xCC30` works!

---

## 📋 Recommended Implementation (Conservative)

### Option 1: Use Existing Count (Safest ✅)

```python
# Already confirmed working:
enemy_count = pyboy_bridge.get_memory(0xCC30)

# Add similar addresses for NPCs/items if they exist:
npc_count = pyboy_bridge.get_memory(0xCC31)  # HYPOTHESIS - needs testing
item_count = pyboy_bridge.get_memory(0xCC32)  # HYPOTHESIS - needs testing
```

### Option 2: Entity Type Checking (Conservative)

```python
# Check entity type at known base addresses:
npc_active = 1 if pyboy_bridge.get_memory(0xD140) != 0x00 else 0
enemy_active = 1 if pyboy_bridge.get_memory(0xD180) != 0x00 else 0
item_active = 1 if pyboy_bridge.get_memory(0xD080) != 0x00 else 0
```

### Option 3: Array Scanning (Most Accurate, Higher CPU)

```python
def count_active_entities(pyboy_bridge, base, count, stride=64):
    """Count active entities in an array."""
    active = 0
    for i in range(count):
        if pyboy_bridge.get_memory(base + i * stride) != 0x00:
            active += 1
    return active

enemy_count = count_active_entities(pyboy_bridge, 0xD180, 8)  # Assume 8 enemy slots
npc_count = count_active_entities(pyboy_bridge, 0xD140, 4)    # Assume 4 NPC slots
```

---

## 🎯 Final Recommendation

**Start with Option 1** (existing count address):

1. Use `0xCC30` for enemy count (already confirmed working)
2. Test adjacent addresses (`0xCC31`, `0xCC32`, etc.) for NPC/item counts
3. If not found, fall back to **Option 2** (entity type checking)
4. Implement **Option 3** only if Options 1 & 2 fail

---

## 📚 References

- **Entity Memory Map**: [wiki.zeldahacking.net/oracle/Objects](https://wiki.zeldahacking.net/oracle/Objects)
- **RAM Map**: [datacrystal.tcrf.net - Oracle of Seasons](https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda:_Oracle_of_Seasons:RAM_map)
- **Similar Games**: NES Zelda RAM map, Link's Awakening disassembly

---

## ✅ Next Steps

1. **Update `zelda_addresses.py`** with new entity addresses
2. **Implement Option 1** (use `0xCC30` + scan adjacent addresses)
3. **Test in local visual mode** (Horon Village = NPCs, overworld = enemies)
4. **Log entity counts** to HUD for validation
5. **Refine based on results**

---

*Research compiled by AI agent analyzing ZeldaHacking.net Wiki and Data Crystal documentation.*

