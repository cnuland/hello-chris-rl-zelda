# Test Suite for Zelda Oracle of Seasons LLM-RL System

This directory contains comprehensive tests for validating the hybrid LLM-RL architecture components.

## Test Files Overview

### `test_simple_env.py` - Basic Functionality Test
**Purpose**: Validates core PyBoy integration and basic game functionality
**Components Tested**:
- PyBoy installation and basic imports
- ROM file loading and validation
- Memory address accessibility
- Save state loading functionality
- Basic game state extraction

**Usage**:
```bash
cd /path/to/hello-chris-rl-llm-zelda
python tests/test_simple_env.py
```

**Expected Results**: 5/5 tests passing in ~0.3 seconds

### `test_advanced_env.py` - Enhanced State Extraction Test
**Purpose**: Validates the complete enhanced state extraction pipeline
**Components Tested**:
- Enhanced game state extraction (16 memory addresses)
- Player position and health tracking
- LLM prompt generation
- Visual data processing and compression
- Structured JSON state creation

**Usage**:
```bash
cd /path/to/hello-chris-rl-llm-zelda
python tests/test_advanced_env.py
```

**Expected Results**: All components working in ~0.13 seconds

### `test_visual_integration.py` - Visual Processing Test
**Purpose**: Validates visual processing pipeline and compression techniques
**Components Tested**:
- Screen array capture from PyBoy
- Visual encoder initialization
- Multiple compression modes (RGB, grayscale, 4-bit, bit-packed)
- Performance benchmarking of different compression techniques

**Usage**:
```bash
cd /path/to/hello-chris-rl-llm-zelda
python tests/test_visual_integration.py
```

### `test_local_env.py` - Full Pipeline Test (Legacy)
**Purpose**: Comprehensive test of the full Gymnasium environment
**Status**: Legacy test file, may have import dependency issues
**Recommendation**: Use `test_simple_env.py` and `test_advanced_env.py` instead

## Test Execution Order

For new setups, run tests in this order:

1. **Basic Validation**:
   ```bash
   python tests/test_simple_env.py
   ```
   Ensures PyBoy, ROM, and basic functionality work

2. **Advanced Features**:
   ```bash
   python tests/test_advanced_env.py
   ```
   Validates enhanced state extraction and LLM integration

3. **Visual Processing** (optional):
   ```bash
   python tests/test_visual_integration.py
   ```
   Tests visual compression pipeline

## Expected Performance

- **test_simple_env.py**: ~0.31 seconds, 5/5 tests passing
- **test_advanced_env.py**: ~0.13 seconds, all components working
- **test_visual_integration.py**: ~2-5 seconds, depending on compression tests

## Requirements

- Python 3.8+
- PyBoy 2.x
- NumPy
- ROM file: `roms/zelda_oracle_of_seasons.gbc`
- Save state file: `roms/zelda_oracle_of_seasons.gbc.state` (optional)

## Troubleshooting

**Import Errors**: Ensure you're running from the project root directory
**ROM Not Found**: Place the ROM file in `roms/zelda_oracle_of_seasons.gbc`
**PyBoy Issues**: Check PyBoy installation: `pip install pyboy`
**Memory Address Failures**: Verify ROM integrity and save state compatibility

## Test Output Files

Tests may generate temporary output files:
- `test_memory_results.json` - Memory validation results
- `test_advanced_results.json` - Enhanced state extraction results

These files are automatically cleaned up or can be safely deleted.

## Integration with CI/CD

For automated testing, run:
```bash
# Basic smoke test
python tests/test_simple_env.py && echo "Basic tests passed"

# Full validation
python tests/test_advanced_env.py && echo "Advanced tests passed"
```

Both tests are designed to work without complex dependencies and provide clear pass/fail indicators.
