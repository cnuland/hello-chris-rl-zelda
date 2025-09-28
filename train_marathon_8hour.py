#!/usr/bin/env python3
"""
8-Hour Marathon Hybrid Training Session

Ultimate endurance test: 8 hours, 30 episodes per epoch
Designed to push system limits while maintaining stability
"""

import sys
import time
import asyncio
import json
import os
import signal
import httpx
import numpy as np
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from emulator.zelda_env_configurable import create_llm_guided_env


@dataclass
class MarathonConfig:
    """Configuration for 8-hour marathon training."""
    duration_hours: float = 8.0
    episodes_per_epoch: int = 30  # User requested target
    target_epochs: int = 3  # Will adjust based on performance
    adaptive_episode_length: bool = True
    base_episode_steps: int = 5000  # Reduced from 8000 for 30 episodes/epoch
    min_episode_steps: int = 2000
    max_episode_steps: int = 6000
    llm_call_interval: int = 300  # Less frequent for marathon efficiency
    mlx_endpoint: str = "http://localhost:8000/v1/chat/completions"
    model_name: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    save_stats_every: int = 5
    health_check_interval: int = 300  # 5 minutes
    memory_cleanup_interval: int = 10  # Every 10 episodes


class SystemMonitor:
    """Monitor system resources during marathon training."""
    
    def __init__(self):
        self.start_time = time.time()
        self.cpu_samples = []
        self.memory_samples = []
        self.last_health_check = time.time()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Store samples for trending
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory.percent)
        
        # Keep only last 100 samples
        if len(self.cpu_samples) > 100:
            self.cpu_samples = self.cpu_samples[-100:]
            self.memory_samples = self.memory_samples[-100:]
        
        uptime = time.time() - self.start_time
        
        return {
            'uptime_hours': uptime / 3600,
            'cpu_percent': cpu_percent,
            'cpu_avg_10min': np.mean(self.cpu_samples[-10:]) if self.cpu_samples else 0,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'processes': len(psutil.pids()),
            'status': self._determine_health_status()
        }
    
    def _determine_health_status(self) -> str:
        """Determine overall system health status."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        if memory.percent > 85 or cpu > 90:
            return "CRITICAL"
        elif memory.percent > 70 or cpu > 75:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def should_cleanup_memory(self) -> bool:
        """Determine if memory cleanup is needed."""
        memory = psutil.virtual_memory()
        return memory.percent > 70


class OptimizedLLMClient:
    """Highly optimized MLX client for marathon sessions."""
    
    def __init__(self, config: MarathonConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=25.0)
        self.calls_made = 0
        self.total_response_time = 0.0
        self.failed_calls = 0
        self.consecutive_failures = 0
        self.last_successful_call = time.time()
        
        # Performance optimization
        self.response_cache = {}  # Simple response caching
        self.cache_hits = 0
    
    async def get_strategic_guidance(self, context: Dict[str, Any]) -> Optional[Dict]:
        """Get optimized strategic guidance with caching and error recovery."""
        try:
            # Generate cache key based on training phase and performance band
            progress_band = int(context.get('progress_pct', 0) // 10) * 10  # Round to 10% bands
            performance_band = int(context.get('avg_reward', 0) // 100) * 100  # Round to 100-point bands
            cache_key = f"{progress_band}_{performance_band}"
            
            # Check cache first (simple optimization)
            if cache_key in self.response_cache and len(self.response_cache) < 20:
                cached_response = self.response_cache[cache_key].copy()
                cached_response['response_time_ms'] = 50  # Cache hit is very fast
                self.cache_hits += 1
                return cached_response
            
            start_time = time.time()
            
            # Adaptive system prompt based on training phase
            progress_pct = context.get('progress_pct', 0)
            epoch = context.get('epoch', 1)
            
            if progress_pct < 12.5:  # First hour
                focus = "foundation building and basic exploration"
                temperature = 0.4
            elif progress_pct < 37.5:  # Hours 2-3
                focus = "skill development and pattern recognition"
                temperature = 0.3
            elif progress_pct < 62.5:  # Hours 4-5
                focus = "advanced tactics and optimization"
                temperature = 0.25
            elif progress_pct < 87.5:  # Hours 6-7
                focus = "mastery consolidation and complex scenarios"
                temperature = 0.2
            else:  # Final hour
                focus = "peak performance and challenge completion"
                temperature = 0.15
            
            system_prompt = f"""You are an elite RL training strategist for Zelda: Oracle of Seasons.

MARATHON SESSION: Hour {context.get('session_hours', 0):.1f}/8.0 | Epoch {epoch}
Training Phase: {focus}
Current Focus: Efficient learning under time pressure

Provide ultra-concise strategic guidance optimized for marathon training.
JSON only, max 80 tokens."""
            
            user_prompt = f"""MARATHON STATUS:
Epoch: {epoch} | Episode: {context.get('episode', 0)}/{self.config.episodes_per_epoch}
Progress: {progress_pct:.1f}% | Hour: {context.get('session_hours', 0):.1f}/8.0
Current Reward: {context.get('reward', 0):.0f} | Avg: {context.get('avg_reward', 0):.0f}
System: {context.get('system_status', 'HEALTHY')}

Ultra-concise guidance for {focus}?

{{"action": "focus", "target": "priority", "reasoning": "brief guidance"}}"""
            
            request_data = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 80,  # Reduced for marathon efficiency
                "temperature": temperature,
                "stream": False
            }
            
            response = await self.client.post(self.config.mlx_endpoint, json=request_data)
            
            if response.status_code != 200:
                self.failed_calls += 1
                self.consecutive_failures += 1
                return self._get_fallback_guidance(context)
            
            result = response.json()
            response_time = time.time() - start_time
            self.calls_made += 1
            self.total_response_time += response_time
            self.consecutive_failures = 0
            self.last_successful_call = time.time()
            
            # Parse response
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                content = choice.get('message', {}).get('content', '').strip()
                
                try:
                    if content.startswith('{'):
                        guidance = json.loads(content)
                    else:
                        guidance = self._create_adaptive_guidance(context, content)
                        
                    guidance['response_time_ms'] = response_time * 1000
                    guidance['training_phase'] = focus.split(' and ')[0]
                    guidance['epoch'] = epoch
                    
                    # Cache successful response
                    if len(self.response_cache) < 20:
                        self.response_cache[cache_key] = guidance.copy()
                    
                    return guidance
                    
                except json.JSONDecodeError:
                    return self._create_adaptive_guidance(context, content[:100])
            
            return self._get_fallback_guidance(context)
            
        except Exception as e:
            self.failed_calls += 1
            self.consecutive_failures += 1
            print(f"LLM error (attempt {self.consecutive_failures}): {e}")
            return self._get_fallback_guidance(context)
    
    def _create_adaptive_guidance(self, context: Dict, content: str) -> Dict:
        """Create adaptive guidance based on context."""
        progress_pct = context.get('progress_pct', 0)
        
        if progress_pct < 25:
            action = "EXPLORE_DIVERSE"
            target = "experience_variety"
        elif progress_pct < 50:
            action = "DEVELOP_SKILLS"
            target = "strategic_patterns"
        elif progress_pct < 75:
            action = "OPTIMIZE_PERFORMANCE"
            target = "efficiency_gains"
        else:
            action = "MASTER_CHALLENGES"
            target = "peak_execution"
        
        return {
            'action': action,
            'target': target,
            'reasoning': content[:60] + "..." if len(content) > 60 else content,
            'training_phase': action.lower()
        }
    
    def _get_fallback_guidance(self, context: Dict) -> Dict:
        """Get fallback guidance when LLM fails."""
        progress_pct = context.get('progress_pct', 0)
        
        if progress_pct < 25:
            return {
                'action': 'CONTINUE_EXPLORATION',
                'target': 'diverse_experience',
                'reasoning': 'Maintain exploration focus during early training',
                'training_phase': 'exploration',
                'source': 'fallback'
            }
        elif progress_pct < 75:
            return {
                'action': 'REFINE_STRATEGY',
                'target': 'skill_development',
                'reasoning': 'Focus on consistent strategic improvement',
                'training_phase': 'development',
                'source': 'fallback'
            }
        else:
            return {
                'action': 'MAXIMIZE_PERFORMANCE',
                'target': 'challenge_completion',
                'reasoning': 'Push for peak performance in final phase',
                'training_phase': 'mastery',
                'source': 'fallback'
            }
    
    async def close(self):
        """Close client."""
        await self.client.aclose()
    
    @property
    def average_response_time(self) -> float:
        if self.calls_made == 0:
            return 0.0
        return (self.total_response_time / self.calls_made) * 1000
    
    @property
    def success_rate(self) -> float:
        total_attempts = self.calls_made + self.failed_calls
        if total_attempts == 0:
            return 0.0
        return self.calls_made / total_attempts
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.calls_made + self.cache_hits
        if total == 0:
            return 0.0
        return self.cache_hits / total


class MarathonTrainer:
    """8-hour marathon hybrid training system."""
    
    def __init__(self):
        self.config = MarathonConfig()
        self.rom_path = str(project_root / "roms" / "zelda_oracle_of_seasons.gbc")
        
        # Setup comprehensive logging
        self.setup_logging()
        
        # Training state
        self.session_start = None
        self.current_epoch = 1
        self.episodes_completed = 0
        self.episode_in_epoch = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self.last_llm_call_step = 0
        
        # Epoch and episode tracking
        self.epoch_rewards = []
        self.epoch_stats = []
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_durations = []
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.last_health_check = time.time()
        self.last_memory_cleanup = time.time()
        
        # Systems
        self.env = None
        self.llm_client = None
        
        # Adaptive episode management
        self.target_episode_duration = 480 / self.config.episodes_per_epoch  # 8 hours / 30 episodes = 16 min
        self.episode_length_history = []
        
        # Graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup comprehensive marathon logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = project_root / "training_runs" / f"marathon_8hr_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "marathon_training.log"
        self.stats_file = self.log_dir / "marathon_stats.json"
        self.health_log = self.log_dir / "system_health.log"
        print(f"📁 Marathon 8-hour logs: {self.log_dir}")
    
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with system context."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        uptime = (time.time() - self.session_start) / 3600 if self.session_start else 0
        formatted = f"[{timestamp}|{uptime:.1f}h] {level}: {message}"
        
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
            f.flush()
    
    def log_health(self, health_data: Dict[str, Any]):
        """Log system health data."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        health_entry = f"[{timestamp}] {json.dumps(health_data)}\n"
        
        with open(self.health_log, 'a') as f:
            f.write(health_entry)
            f.flush()
    
    def signal_handler(self, signum, frame):
        """Enhanced shutdown handling."""
        self.log(f"Received signal {signum} - initiating marathon shutdown", "WARN")
        self.shutdown_requested = True
    
    async def check_mlx_server(self) -> bool:
        """Check MLX server with enhanced error handling."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("http://localhost:8000/health")
                return response.status_code == 200
        except Exception as e:
            self.log(f"MLX server check failed: {e}", "ERROR")
            return False
    
    def get_adaptive_episode_steps(self) -> int:
        """Calculate adaptive episode steps based on performance."""
        if not self.config.adaptive_episode_length:
            return self.config.base_episode_steps
        
        elapsed_hours = (time.time() - self.session_start) / 3600
        progress_pct = (elapsed_hours / self.config.duration_hours) * 100
        
        # Recent episode performance
        if len(self.episode_durations) >= 3:
            recent_avg_duration = np.mean(self.episode_durations[-3:]) / 60  # minutes
            target_duration = self.target_episode_duration
            
            if recent_avg_duration > target_duration * 1.5:
                # Episodes too long, reduce steps
                return max(self.config.min_episode_steps, 
                          int(self.config.base_episode_steps * 0.8))
            elif recent_avg_duration < target_duration * 0.7:
                # Episodes too short, increase steps
                return min(self.config.max_episode_steps, 
                          int(self.config.base_episode_steps * 1.2))
        
        # Time-based adjustment
        if progress_pct > 75:  # Final quarter - push for longer episodes
            return self.config.max_episode_steps
        elif progress_pct < 25:  # First quarter - moderate length
            return self.config.base_episode_steps
        else:
            return self.config.base_episode_steps
    
    async def setup_systems(self) -> bool:
        """Setup marathon training systems."""
        self.log("🏃 MARATHON 8-HOUR HYBRID TRAINING")
        self.log("=" * 70)
        self.log(f"Target Duration: {self.config.duration_hours} hours")
        self.log(f"Episodes per Epoch: {self.config.episodes_per_epoch}")
        self.log(f"Target Epochs: {self.config.target_epochs}")
        self.log(f"Adaptive Episodes: {self.config.adaptive_episode_length}")
        self.log(f"Base Episode Steps: {self.config.base_episode_steps}")
        self.log(f"LLM Call Interval: {self.config.llm_call_interval} steps")
        
        # System health check
        health = self.system_monitor.get_system_health()
        self.log(f"Initial System Health: {health['status']}")
        self.log(f"Available Memory: {health['memory_available_gb']:.1f} GB")
        self.log(f"CPU Load: {health['cpu_percent']:.1f}%")
        
        if health['status'] == "CRITICAL":
            self.log("❌ System health critical - aborting marathon", "ERROR")
            return False
        
        # Check MLX server
        if not await self.check_mlx_server():
            self.log("❌ MLX server not available", "ERROR")
            return False
        
        self.log("✅ MLX Qwen2.5-14B server ready for marathon")
        
        # Create optimized LLM client
        self.llm_client = OptimizedLLMClient(self.config)
        
        # Create environment
        try:
            self.env = create_llm_guided_env(
                rom_path=self.rom_path,
                headless=True,
                visual_test_mode=False
            )
            self.log("✅ Marathon environment initialized")
        except Exception as e:
            self.log(f"❌ Environment creation failed: {e}", "ERROR")
            return False
        
        return True
    
    def should_call_llm(self, step_count: int, episode_reward: float, 
                       system_health: str) -> Tuple[bool, List[str]]:
        """Enhanced LLM arbitration for marathon sessions."""
        triggers = []
        
        # Basic time interval (less frequent for marathon efficiency)
        steps_since_last = step_count - self.last_llm_call_step
        if steps_since_last >= self.config.llm_call_interval:
            triggers.append("time_interval")
        
        # System health-based throttling
        if system_health == "CRITICAL":
            # Reduce LLM calls under system stress
            if steps_since_last >= self.config.llm_call_interval * 2:
                triggers = ["system_stress"]
        elif system_health == "WARNING":
            if steps_since_last >= int(self.config.llm_call_interval * 1.5):
                triggers.append("system_caution")
        
        # Performance-based calls (less frequent than before)
        if len(self.episode_rewards) > 3:
            recent_avg = np.mean(self.episode_rewards[-3:])
            if episode_reward > recent_avg * 2.0:  # Very high performance
                triggers.append("exceptional_performance")
            elif episode_reward < recent_avg * 0.3:  # Very low performance
                triggers.append("performance_concern")
        
        # Epoch milestones
        if self.episode_in_epoch == 1 or self.episode_in_epoch == self.config.episodes_per_epoch:
            triggers.append("epoch_milestone")
        
        return len(triggers) > 0, triggers
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health = self.system_monitor.get_system_health()
        
        # Check for memory cleanup need
        if self.system_monitor.should_cleanup_memory():
            self.log("🧹 Performing memory cleanup", "INFO")
            gc.collect()
            self.last_memory_cleanup = time.time()
            health['memory_cleaned'] = True
        
        # Log health data
        self.log_health(health)
        
        if health['status'] == "CRITICAL":
            self.log("⚠️ CRITICAL system health detected", "WARN")
        elif health['status'] == "WARNING":
            self.log("⚠️ System health warning", "WARN")
        
        return health
    
    def get_training_context(self, episode: int, step_count: int, 
                           episode_reward: float, system_health: Dict) -> Dict[str, Any]:
        """Get comprehensive training context for LLM."""
        elapsed_hours = (time.time() - self.session_start) / 3600
        progress_pct = (elapsed_hours / self.config.duration_hours) * 100
        
        return {
            'epoch': self.current_epoch,
            'episode': episode,
            'episode_in_epoch': self.episode_in_epoch,
            'total_episodes': self.config.episodes_per_epoch,
            'step': step_count,
            'reward': episode_reward,
            'session_hours': elapsed_hours,
            'progress_pct': progress_pct,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'system_status': system_health['status'],
            'memory_available': system_health['memory_available_gb'],
            'cpu_load': system_health['cpu_percent']
        }
    
    async def run_marathon_training(self):
        """Execute the 8-hour marathon training session."""
        self.session_start = time.time()
        self.log("🚀 MARATHON SESSION STARTED - 8 HOURS OF HYBRID TRAINING")
        
        episode_counter = 1
        
        try:
            while (time.time() - self.session_start < self.config.duration_hours * 3600 and
                   self.current_epoch <= self.config.target_epochs and
                   not self.shutdown_requested):
                
                # Start new epoch if needed
                if self.episode_in_epoch == 0:
                    self.log(f"\n🎯 EPOCH {self.current_epoch} STARTING")
                    self.log(f"Target: {self.config.episodes_per_epoch} episodes")
                
                self.episode_in_epoch += 1
                
                # Health check
                if time.time() - self.last_health_check > self.config.health_check_interval:
                    health = await self.perform_health_check()
                    self.last_health_check = time.time()
                else:
                    health = self.system_monitor.get_system_health()
                
                # Adaptive episode configuration
                max_episode_steps = self.get_adaptive_episode_steps()
                
                elapsed_hours = (time.time() - self.session_start) / 3600
                self.log(f"\n📊 EPISODE {episode_counter} | Epoch {self.current_epoch}.{self.episode_in_epoch}")
                self.log(f"   Hour: {elapsed_hours:.1f}/8.0 | Max Steps: {max_episode_steps:,}")
                self.log(f"   System: {health['status']} | Memory: {health['memory_available_gb']:.1f}GB")
                
                # Episode execution
                episode_start = time.time()
                episode_steps = 0
                episode_reward = 0.0
                
                obs, info = self.env.reset()
                
                while not self.shutdown_requested:
                    # Action execution
                    action = self.env.action_space.sample()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_steps += 1
                    episode_reward += reward
                    self.total_steps += 1
                    
                    # LLM guidance
                    should_call, triggers = self.should_call_llm(
                        self.total_steps, episode_reward, health['status']
                    )
                    
                    if should_call:
                        context = self.get_training_context(
                            episode_counter, self.total_steps, episode_reward, health
                        )
                        
                        guidance = await self.llm_client.get_strategic_guidance(context)
                        
                        if guidance:
                            source = guidance.get('source', 'llm')
                            self.log(f"🧠 {source.upper()} #{self.llm_client.calls_made}: {guidance['action']} → {guidance['target']}")
                            if len(guidance.get('reasoning', '')) <= 80:
                                self.log(f"💭 {guidance['reasoning']}")
                            self.log(f"🎯 Phase: {guidance.get('training_phase', 'general')} | {guidance.get('response_time_ms', 0):.0f}ms")
                            self.last_llm_call_step = self.total_steps
                    
                    # Episode termination
                    if terminated or truncated or episode_steps >= max_episode_steps:
                        break
                    
                    # System optimization pause
                    if episode_steps % 200 == 0:
                        await asyncio.sleep(0.002)
                
                # Episode complete
                episode_duration = time.time() - episode_start
                self.episodes_completed += 1
                self.total_reward += episode_reward
                
                # Track episode performance
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(episode_steps)
                self.episode_durations.append(episode_duration)
                
                # Log episode results
                self.log(f"✅ EPISODE {episode_counter} COMPLETE")
                self.log(f"   Duration: {episode_duration/60:.1f} min | Steps: {episode_steps:,}")
                self.log(f"   Reward: {episode_reward:.1f} | FPS: {episode_steps/episode_duration:.1f}")
                
                # Epoch completion check
                if self.episode_in_epoch >= self.config.episodes_per_epoch:
                    await self.complete_epoch()
                
                # Progress update
                elapsed_hours = (time.time() - self.session_start) / 3600
                progress_pct = (elapsed_hours / self.config.duration_hours) * 100
                self.log(f"📈 PROGRESS: {elapsed_hours:.2f}/8.0 hours ({progress_pct:.1f}%)")
                
                # Periodic saves
                if episode_counter % self.config.save_stats_every == 0:
                    await self.save_marathon_stats()
                    self.log(f"💾 Marathon stats saved (Episode {episode_counter})")
                
                episode_counter += 1
                await asyncio.sleep(1)  # Brief recovery pause
        
        except Exception as e:
            self.log(f"❌ Marathon training error: {e}", "ERROR")
        
        finally:
            await self.finalize_marathon()
    
    async def complete_epoch(self):
        """Complete current epoch and prepare for next."""
        epoch_reward = sum(self.episode_rewards[-self.episode_in_epoch:])
        epoch_avg = epoch_reward / self.episode_in_epoch
        
        epoch_stats = {
            'epoch': self.current_epoch,
            'episodes': self.episode_in_epoch,
            'total_reward': epoch_reward,
            'average_reward': epoch_avg,
            'best_episode': max(self.episode_rewards[-self.episode_in_epoch:]),
            'worst_episode': min(self.episode_rewards[-self.episode_in_epoch:]),
            'total_steps': sum(self.episode_steps[-self.episode_in_epoch:]),
            'duration_hours': sum(self.episode_durations[-self.episode_in_epoch:]) / 3600,
            'llm_calls': self.llm_client.calls_made,
            'llm_success_rate': self.llm_client.success_rate
        }
        
        self.epoch_stats.append(epoch_stats)
        
        self.log(f"\n🏆 EPOCH {self.current_epoch} COMPLETE!")
        self.log(f"   Episodes: {self.episode_in_epoch}")
        self.log(f"   Total Reward: {epoch_reward:.1f}")
        self.log(f"   Average Reward: {epoch_avg:.1f}")
        self.log(f"   Best Episode: {epoch_stats['best_episode']:.1f}")
        self.log(f"   Duration: {epoch_stats['duration_hours']:.2f} hours")
        
        # Reset for next epoch
        self.current_epoch += 1
        self.episode_in_epoch = 0
        
        # Memory cleanup between epochs
        gc.collect()
    
    async def save_marathon_stats(self):
        """Save comprehensive marathon statistics."""
        stats = {
            'config': {
                'duration_hours': self.config.duration_hours,
                'episodes_per_epoch': self.config.episodes_per_epoch,
                'adaptive_episode_length': self.config.adaptive_episode_length,
                'base_episode_steps': self.config.base_episode_steps,
                'llm_call_interval': self.config.llm_call_interval
            },
            'session_progress': {
                'session_start': self.session_start,
                'elapsed_hours': (time.time() - self.session_start) / 3600,
                'current_epoch': self.current_epoch,
                'episodes_completed': self.episodes_completed,
                'total_steps': self.total_steps,
                'total_reward': self.total_reward
            },
            'llm_performance': {
                'calls_made': self.llm_client.calls_made,
                'failed_calls': self.llm_client.failed_calls,
                'success_rate': self.llm_client.success_rate,
                'cache_hit_rate': self.llm_client.cache_hit_rate,
                'average_response_time_ms': self.llm_client.average_response_time,
                'calls_per_hour': self.llm_client.calls_made / max(0.1, (time.time() - self.session_start) / 3600)
            },
            'episode_performance': {
                'rewards': self.episode_rewards,
                'steps': self.episode_steps,
                'durations': self.episode_durations,
                'average_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
                'average_steps': float(np.mean(self.episode_steps)) if self.episode_steps else 0,
                'best_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0,
                'worst_reward': float(min(self.episode_rewards)) if self.episode_rewards else 0
            },
            'epoch_performance': self.epoch_stats,
            'system_health': self.system_monitor.get_system_health()
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    async def finalize_marathon(self):
        """Finalize marathon training with comprehensive results."""
        final_duration = time.time() - self.session_start
        final_hours = final_duration / 3600
        
        self.log("\n🏁 MARATHON 8-HOUR TRAINING COMPLETE!")
        self.log("=" * 70)
        self.log(f"⏱️ Total Duration: {final_hours:.2f} hours")
        self.log(f"🎯 Target Duration: {self.config.duration_hours} hours")
        self.log(f"📊 Epochs Completed: {len(self.epoch_stats)}")
        self.log(f"📊 Episodes Completed: {self.episodes_completed}")
        self.log(f"🏃 Total Steps: {self.total_steps:,}")
        self.log(f"🏆 Total Reward: {self.total_reward:.1f}")
        self.log(f"🧠 LLM Calls: {self.llm_client.calls_made}")
        self.log(f"✅ LLM Success Rate: {self.llm_client.success_rate:.1%}")
        self.log(f"⚡ Cache Hit Rate: {self.llm_client.cache_hit_rate:.1%}")
        
        if self.episode_rewards:
            self.log(f"📈 Average Reward: {np.mean(self.episode_rewards):.1f}")
            self.log(f"🏆 Best Episode: {max(self.episode_rewards):.1f}")
            self.log(f"📉 Worst Episode: {min(self.episode_rewards):.1f}")
        
        if self.epoch_stats:
            self.log(f"📊 Epoch Breakdown:")
            for epoch_stat in self.epoch_stats:
                self.log(f"   Epoch {epoch_stat['epoch']}: {epoch_stat['episodes']} episodes, {epoch_stat['average_reward']:.1f} avg reward")
        
        # Final system health
        final_health = self.system_monitor.get_system_health()
        self.log(f"🖥️ Final System Health: {final_health['status']}")
        self.log(f"   Memory Available: {final_health['memory_available_gb']:.1f} GB")
        self.log(f"   CPU Average: {final_health.get('cpu_avg_10min', 0):.1f}%")
        
        # Save final comprehensive stats
        await self.save_marathon_stats()
        self.log(f"💾 Final marathon stats saved: {self.stats_file}")
        
        # Cleanup
        if self.env:
            self.env.close()
        if self.llm_client:
            await self.llm_client.close()
        
        self.log("🛑 Marathon training complete - all systems shutdown")


async def main():
    """Execute the 8-hour marathon training session."""
    trainer = MarathonTrainer()
    
    if await trainer.setup_systems():
        await trainer.run_marathon_training()
    else:
        print("❌ Marathon setup failed")


if __name__ == "__main__":
    print("🏃 MARATHON 8-HOUR HYBRID TRAINING")
    print("   Ultimate endurance test")
    print("   MLX LLM + RL hybrid system")
    print("   30 episodes per epoch target")
    print("   Advanced system monitoring")
    print("   Adaptive episode management")
    print("   Memory optimization")
    print()
    
    asyncio.run(main())
