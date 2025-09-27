"""
Local LLM Planner for Zelda Oracle of Seasons

Optimized planner client designed specifically for local vLLM servers.
Takes advantage of low latency and high availability of local inference.
"""

import json
import httpx
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from pathlib import Path

from .macro_actions import MacroAction, create_macro_from_planner_output


@dataclass
class LocalLLMConfig:
    """Configuration optimized for MLX local LLM server."""
    endpoint_url: str = "http://localhost:8000/v1/chat/completions"
    model_name: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    max_tokens: int = 100              # Reduced for speed
    temperature: float = 0.3           # More deterministic for local
    timeout: float = 10.0              # Account for 1-2 second response time
    max_retries: int = 2               # Fewer retries needed
    
    # MLX optimization settings
    enable_fast_mode: bool = True      # Use shorter prompts
    cache_responses: bool = True       # Cache similar states
    expected_latency_ms: int = 1500    # ~1.3 seconds based on test
    min_call_interval_ms: int = 500    # Minimum between calls for MLX


class LocalZeldaPlanner:
    """LLM planner optimized for local vLLM inference."""

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        """Initialize local LLM planner."""
        self.config = config or LocalLLMConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.logger = logging.getLogger(__name__)
        
        # Local optimization features
        self.response_cache = {}
        self.last_call_time = 0
        self.call_count = 0
        self.total_latency = 0
        
        # Load optimized prompts
        self.system_prompt = self._load_optimized_system_prompt()
        self.fast_prompt_template = self._load_fast_prompt_template()

    def _load_optimized_system_prompt(self) -> str:
        """Load system prompt optimized for local LLM performance."""
        # Try to load from config file first
        try:
            config_path = Path(__file__).parent.parent / "configs" / "planner_prompt.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('system_prompt', '')
        except Exception as e:
            self.logger.warning(f"Could not load system prompt from config: {e}")
        
        # Fallback to optimized built-in prompt for MLX
        return """You are an expert AI playing The Legend of Zelda: Oracle of Seasons. Analyze the game state and choose the BEST action.

Available Actions:
- MOVE_TO: Move to a specific location
- EXPLORE_ROOM: Explore the current room systematically  
- ATTACK_ENEMY: Attack a nearby enemy
- COLLECT_ITEM: Collect an item or rupee
- USE_ITEM: Use an inventory item
- ENTER_DOOR: Enter a door or passage

ALWAYS respond with valid JSON only:
{"action": "ACTION_NAME", "target": "brief target description", "reasoning": "why this action"}

Priority: Safety first, then exploration, then combat. Be decisive and strategic."""

    def _load_fast_prompt_template(self) -> str:
        """Load fast prompt template for local processing."""
        return """Current state:
Health: {health}/{max_health} hearts
Room: {room} 
Position: ({x}, {y})
Items: {items}

Nearby: {nearby_elements}

Action needed (JSON only):"""

    def _create_cache_key(self, game_state: Dict[str, Any]) -> str:
        """Create cache key for similar game states."""
        # Create simplified key based on core state elements
        player = game_state.get('player', {})
        key_elements = {
            'room': player.get('room', 0),
            'health_ratio': round(player.get('health', 3) / max(1, player.get('max_health', 3)), 1),
            'position_area': (player.get('x', 0) // 20, player.get('y', 0) // 20),  # Grid-based position
            'has_items': len(game_state.get('resources', {}).get('inventory', [])) > 0
        }
        return str(hash(frozenset(key_elements.items())))

    async def _check_server_health(self) -> bool:
        """Check if the local LLM server is responsive."""
        try:
            response = await self.client.get(
                self.config.endpoint_url.replace('/generate', '/health'),
                timeout=2.0
            )
            return response.status_code == 200
        except Exception:
            return False

    def _should_throttle_request(self) -> bool:
        """Check if we should throttle the request based on timing."""
        current_time = time.time() * 1000  # Convert to milliseconds
        time_since_last = current_time - self.last_call_time
        return time_since_last < self.config.min_call_interval_ms

    async def get_plan(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Get plan from local LLM with optimizations."""
        
        # Check throttling
        if self._should_throttle_request():
            self.logger.debug("Request throttled - too frequent calls")
            return None
        
        # Check cache if enabled
        cache_key = None
        if self.config.cache_responses:
            cache_key = self._create_cache_key(game_state)
            if cache_key in self.response_cache:
                self.logger.debug("Using cached response")
                return self.response_cache[cache_key]

        # Check server health
        if not await self._check_server_health():
            self.logger.error("Local LLM server not responding")
            return None

        try:
            start_time = time.time()
            
            # Create optimized prompt
            if self.config.enable_fast_mode:
                prompt = self._create_fast_prompt(game_state)
            else:
                prompt = self._create_user_prompt(game_state)

            # Make request to local MLX server (OpenAI Chat Completions API)
            request_data = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }

            response = await self.client.post(
                self.config.endpoint_url,
                json=request_data
            )

            if response.status_code != 200:
                self.logger.error(f"LLM request failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            
            # Extract response text (OpenAI Chat Completions format)
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice:
                    response_text = choice['message'].get('content', '').strip()
                else:
                    response_text = choice.get('text', '').strip()
            else:
                self.logger.error(f"Unexpected response format: {result}")
                return None

            # Update performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.call_count += 1
            self.total_latency += latency_ms
            self.last_call_time = time.time() * 1000

            # Cache successful response
            if cache_key and response_text:
                self.response_cache[cache_key] = response_text

            self.logger.debug(f"LLM response in {latency_ms:.1f}ms: {response_text[:100]}...")
            
            return response_text

        except Exception as e:
            self.logger.error(f"Error calling local LLM: {e}")
            return None

    def _create_fast_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create fast, concise prompt for local processing."""
        player = game_state.get('player', {})
        resources = game_state.get('resources', {})
        
        # Get nearby elements (simplified)
        nearby_elements = []
        entities = game_state.get('entities', {})
        if entities.get('enemies'):
            nearby_elements.append(f"enemies: {len(entities['enemies'])}")
        if entities.get('items'):
            nearby_elements.append(f"items: {len(entities['items'])}")
        if entities.get('npcs'):
            nearby_elements.append(f"npcs: {len(entities['npcs'])}")
        
        return self.fast_prompt_template.format(
            health=player.get('health', 3),
            max_health=player.get('max_health', 3),
            room=player.get('room', 0),
            x=player.get('x', 0),
            y=player.get('y', 0),
            items=len(resources.get('inventory', [])),
            nearby_elements=', '.join(nearby_elements) if nearby_elements else 'nothing'
        )

    def _create_user_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create standard user prompt (fallback)."""
        # Simplified version of the full prompt for local processing
        player = game_state.get('player', {})
        
        prompt_parts = [
            f"Current game state:",
            f"Player: {player.get('health', 3)}/{player.get('max_health', 3)} hearts at room {player.get('room', 0)}",
            f"Position: ({player.get('x', 0)}, {player.get('y', 0)})",
        ]
        
        # Add entities if present
        entities = game_state.get('entities', {})
        if entities:
            entity_summary = []
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_summary.append(f"{len(entity_list)} {entity_type}")
            if entity_summary:
                prompt_parts.append(f"Nearby: {', '.join(entity_summary)}")
        
        prompt_parts.append("\nProvide action as JSON:")
        
        return '\n'.join(prompt_parts)

    def get_macro_action(self, plan_text: str) -> Optional[MacroAction]:
        """Convert LLM plan to macro action."""
        try:
            return create_macro_from_planner_output(plan_text)
        except Exception as e:
            self.logger.error(f"Error creating macro from plan: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the local LLM."""
        avg_latency = self.total_latency / max(1, self.call_count)
        return {
            'total_calls': self.call_count,
            'average_latency_ms': avg_latency,
            'cache_hit_rate': len(self.response_cache) / max(1, self.call_count),
            'expected_latency_ms': self.config.expected_latency_ms,
            'performance_ratio': self.config.expected_latency_ms / max(1, avg_latency)
        }

    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Factory function for easy creation
def create_local_planner(config_path: Optional[str] = None) -> LocalZeldaPlanner:
    """Create a local LLM planner with configuration."""
    config = LocalLLMConfig()
    
    # Load config from file if provided
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                planner_config = yaml_config.get('planner', {})
                
                # Update config with YAML values
                for field_name, value in planner_config.items():
                    if hasattr(config, field_name):
                        setattr(config, field_name, value)
                        
        except Exception as e:
            logging.warning(f"Could not load local planner config: {e}")
    
    return LocalZeldaPlanner(config)
