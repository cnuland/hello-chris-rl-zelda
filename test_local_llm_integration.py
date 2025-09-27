#!/usr/bin/env python3
"""
Local LLM Integration Test

Tests the integration between the smart arbitration system and the local vLLM server
running on localhost:8000. Validates performance, latency, and functionality.
"""

import sys
import time
import asyncio
import json
import httpx
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from agents.local_llm_planner import LocalZeldaPlanner, LocalLLMConfig
    from agents.controller import SmartArbitrationTracker, ControllerConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This test requires the local LLM planner implementation")
    sys.exit(1)


class LocalLLMIntegrationTest:
    """Test suite for local LLM integration."""
    
    def __init__(self):
        self.server_url = "http://localhost:8000"
        self.test_results = {}
        
    async def test_server_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity to the local vLLM server."""
        print("🔗 Testing local LLM server connectivity...")
        
        results = {
            'server_reachable': False,
            'health_check': False,
            'generate_endpoint': False,
            'response_time_ms': 0
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test basic connectivity
                start_time = time.time()
                response = await client.get(f"{self.server_url}/health")
                response_time = (time.time() - start_time) * 1000
                
                results['server_reachable'] = True
                results['health_check'] = response.status_code == 200
                results['response_time_ms'] = response_time
                
                print(f"  ✅ Server reachable at {self.server_url}")
                print(f"  ✅ Health check: {response.status_code}")
                print(f"  ⚡ Response time: {response_time:.1f}ms")
                
                # Test generate endpoint with simple request
                test_request = {
                    "prompt": "Test prompt",
                    "max_tokens": 10,
                    "temperature": 0.1
                }
                
                start_time = time.time()
                gen_response = await client.post(
                    f"{self.server_url}/generate",
                    json=test_request
                )
                gen_time = (time.time() - start_time) * 1000
                
                results['generate_endpoint'] = gen_response.status_code == 200
                results['generate_response_time_ms'] = gen_time
                
                if gen_response.status_code == 200:
                    print(f"  ✅ Generate endpoint working: {gen_time:.1f}ms")
                else:
                    print(f"  ⚠️ Generate endpoint issue: {gen_response.status_code}")
                    
        except Exception as e:
            print(f"  ❌ Connection failed: {e}")
            results['error'] = str(e)
            
        return results
    
    async def test_local_planner_integration(self) -> Dict[str, Any]:
        """Test the LocalZeldaPlanner with real server."""
        print("\n🧠 Testing LocalZeldaPlanner integration...")
        
        results = {
            'planner_created': False,
            'response_received': False,
            'response_parsed': False,
            'average_latency_ms': 0,
            'cache_working': False,
            'responses': []
        }
        
        try:
            # Create local planner with optimized config
            config = LocalLLMConfig(
                endpoint_url=f"{self.server_url}/generate",
                max_tokens=128,
                temperature=0.3,
                timeout=5.0,
                enable_fast_mode=True,
                cache_responses=True
            )
            
            planner = LocalZeldaPlanner(config)
            results['planner_created'] = True
            print("  ✅ LocalZeldaPlanner created")
            
            # Test with realistic game state
            test_game_state = {
                'player': {
                    'health': 3,
                    'max_health': 3,
                    'x': 80,
                    'y': 72,
                    'room': 1
                },
                'resources': {
                    'rupees': 10,
                    'inventory': ['wooden_sword']
                },
                'entities': {
                    'enemies': [{'type': 'octorok', 'x': 60, 'y': 80}],
                    'items': [{'type': 'rupee', 'x': 100, 'y': 60}],
                    'npcs': []
                }
            }
            
            # Test multiple requests for latency measurement
            latencies = []
            for i in range(3):
                start_time = time.time()
                response = await planner.get_plan(test_game_state)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if response:
                    results['response_received'] = True
                    results['responses'].append(response[:100])
                    print(f"  ✅ Response {i+1}: {latency:.1f}ms - {response[:50]}...")
                    
                    # Try to parse as JSON
                    try:
                        if '{' in response:
                            json_part = response[response.find('{'):response.rfind('}')+1]
                            parsed = json.loads(json_part)
                            results['response_parsed'] = True
                            print(f"    📋 Parsed action: {parsed.get('action', 'unknown')}")
                    except:
                        print(f"    ⚠️ Response not valid JSON")
                else:
                    print(f"  ❌ No response received for request {i+1}")
                    
                # Brief pause between requests
                await asyncio.sleep(0.1)
            
            if latencies:
                results['average_latency_ms'] = sum(latencies) / len(latencies)
                print(f"  📊 Average latency: {results['average_latency_ms']:.1f}ms")
            
            # Test caching
            cache_start = time.time()
            cached_response = await planner.get_plan(test_game_state)  # Should be cached
            cache_time = (time.time() - cache_start) * 1000
            
            if cache_time < results['average_latency_ms'] / 2:
                results['cache_working'] = True
                print(f"  ✅ Cache working: {cache_time:.1f}ms (much faster)")
            
            # Get performance stats
            stats = planner.get_performance_stats()
            print(f"  📈 Performance stats:")
            print(f"    Total calls: {stats['total_calls']}")
            print(f"    Average latency: {stats['average_latency_ms']:.1f}ms")
            print(f"    Cache hit rate: {stats['cache_hit_rate']:.1%}")
            
            await planner.close()
            
        except Exception as e:
            print(f"  ❌ LocalZeldaPlanner test failed: {e}")
            results['error'] = str(e)
            
        return results
    
    async def test_smart_arbitration_with_local_llm(self) -> Dict[str, Any]:
        """Test smart arbitration system with local LLM."""
        print("\n🎯 Testing Smart Arbitration with Local LLM...")
        
        results = {
            'arbitration_created': False,
            'triggers_detected': [],
            'local_llm_calls': 0,
            'total_response_time': 0,
            'integration_working': False
        }
        
        try:
            # Create smart arbitration config optimized for local LLM
            config = ControllerConfig(
                use_smart_arbitration=True,
                base_planner_frequency=50,  # More frequent for local (faster) LLM
                min_planner_frequency=20,   # Can call more often
                max_planner_frequency=150,  # Less conservative
                trigger_on_new_room=True,
                trigger_on_low_health=True,
                trigger_on_stuck=True,
                trigger_on_npc_interaction=True,
                macro_timeout=50           # Faster timeout for local
            )
            
            tracker = SmartArbitrationTracker(config)
            results['arbitration_created'] = True
            print("  ✅ Smart arbitration tracker created")
            
            # Create local planner
            planner = LocalZeldaPlanner(LocalLLMConfig(
                endpoint_url=f"{self.server_url}/generate",
                max_tokens=64,              # Even faster for testing
                temperature=0.1,            # More deterministic
                enable_fast_mode=True
            ))
            
            # Simulate scenario with various triggers
            scenarios = [
                {
                    'step': 50,
                    'state': {
                        'player': {'health': 3, 'max_health': 3, 'room': 0, 'x': 80, 'y': 72}
                    },
                    'expected_trigger': 'TIME_INTERVAL'
                },
                {
                    'step': 60,
                    'state': {
                        'player': {'health': 3, 'max_health': 3, 'room': 1, 'x': 120, 'y': 90}  # New room
                    },
                    'expected_trigger': 'NEW_ROOM'
                },
                {
                    'step': 80,
                    'state': {
                        'player': {'health': 1, 'max_health': 3, 'room': 1, 'x': 120, 'y': 90},  # Low health
                    },
                    'expected_trigger': 'LOW_HEALTH'
                },
                {
                    'step': 100,
                    'state': {
                        'player': {'health': 1, 'max_health': 3, 'room': 1, 'x': 120, 'y': 90},
                        'dialogue_state': 1  # NPC interaction
                    },
                    'expected_trigger': 'NPC_INTERACTION'
                }
            ]
            
            total_start_time = time.time()
            
            for scenario in scenarios:
                should_call, triggers = tracker.should_call_llm(
                    scenario['step'], 
                    scenario['state']
                )
                
                if should_call:
                    trigger_names = [t.value for t in triggers]
                    results['triggers_detected'].extend(trigger_names)
                    
                    print(f"  🎯 Step {scenario['step']}: Triggers {trigger_names}")
                    
                    # Test actual LLM call
                    start_time = time.time()
                    response = await planner.get_plan(scenario['state'])
                    call_time = (time.time() - start_time) * 1000
                    
                    if response:
                        results['local_llm_calls'] += 1
                        results['total_response_time'] += call_time
                        tracker.record_arbitration_call(scenario['step'])
                        
                        # Assume success if we got a response
                        if len(response.strip()) > 10:
                            tracker.record_arbitration_success()
                            
                        print(f"    ⚡ LLM response in {call_time:.1f}ms: {response[:30]}...")
                    else:
                        print(f"    ❌ No LLM response")
                        
                else:
                    print(f"  ⏭️ Step {scenario['step']}: No triggers")
            
            total_time = (time.time() - total_start_time) * 1000
            
            # Check integration success
            if (results['local_llm_calls'] > 0 and 
                len(results['triggers_detected']) >= 3 and
                results['total_response_time'] < 2000):  # Under 2 seconds total
                results['integration_working'] = True
            
            print(f"  📊 Integration Results:")
            print(f"    LLM calls: {results['local_llm_calls']}")
            print(f"    Triggers detected: {len(results['triggers_detected'])}")
            print(f"    Average response time: {results['total_response_time'] / max(1, results['local_llm_calls']):.1f}ms")
            print(f"    Total test time: {total_time:.1f}ms")
            
            # Get arbitration stats
            stats = tracker.get_arbitration_stats()
            print(f"  🧠 Arbitration Stats:")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Rooms per call: {stats['rooms_per_call']:.1f}")
            
            await planner.close()
            
        except Exception as e:
            print(f"  ❌ Smart arbitration integration test failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def calculate_performance_score(self, all_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall performance score for local LLM integration."""
        
        weights = {
            'connectivity': 0.2,
            'planner_integration': 0.4,
            'arbitration_integration': 0.4
        }
        
        scores = {}
        
        # Connectivity score
        conn = all_results.get('connectivity', {})
        if conn.get('server_reachable') and conn.get('health_check') and conn.get('generate_endpoint'):
            conn_score = 1.0
            if conn.get('response_time_ms', 1000) < 100:
                conn_score = 1.0
            elif conn.get('response_time_ms', 1000) < 500:
                conn_score = 0.8
            else:
                conn_score = 0.6
        else:
            conn_score = 0.0
        scores['connectivity'] = conn_score
        
        # Planner integration score
        planner = all_results.get('planner_integration', {})
        planner_score = 0.0
        if planner.get('planner_created'):
            planner_score += 0.3
        if planner.get('response_received'):
            planner_score += 0.4
        if planner.get('response_parsed'):
            planner_score += 0.3
        if planner.get('average_latency_ms', 1000) < 500:
            planner_score += 0.2
        if planner.get('cache_working'):
            planner_score += 0.2
        scores['planner_integration'] = min(1.0, planner_score)
        
        # Arbitration integration score
        arb = all_results.get('arbitration_integration', {})
        arb_score = 0.0
        if arb.get('arbitration_created'):
            arb_score += 0.2
        if len(arb.get('triggers_detected', [])) >= 3:
            arb_score += 0.4
        if arb.get('local_llm_calls', 0) > 0:
            arb_score += 0.2
        if arb.get('integration_working'):
            arb_score += 0.2
        scores['arbitration_integration'] = arb_score
        
        # Calculate weighted score
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        return overall_score, scores


async def main():
    """Run the complete local LLM integration test."""
    print("🚀 LOCAL LLM INTEGRATION TEST")
    print("=" * 60)
    print("Testing smart arbitration with local vLLM server")
    print()
    
    test_runner = LocalLLMIntegrationTest()
    all_results = {}
    
    try:
        # Test 1: Server connectivity
        all_results['connectivity'] = await test_runner.test_server_connectivity()
        
        if not all_results['connectivity'].get('server_reachable'):
            print("\n❌ Cannot proceed - local LLM server not reachable")
            print("Make sure vLLM is running on http://localhost:8000")
            return False
        
        # Test 2: Local planner integration
        all_results['planner_integration'] = await test_runner.test_local_planner_integration()
        
        # Test 3: Smart arbitration with local LLM
        all_results['arbitration_integration'] = await test_runner.test_smart_arbitration_with_local_llm()
        
        # Calculate performance score
        overall_score, component_scores = test_runner.calculate_performance_score(all_results)
        
        print(f"\n" + "="*60)
        print("📊 LOCAL LLM INTEGRATION TEST RESULTS")
        print("="*60)
        
        for component, score in component_scores.items():
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            print(f"{status} {component.replace('_', ' ').title()}: {score:.0%}")
        
        overall_status = "✅" if overall_score >= 0.8 else "⚠️" if overall_score >= 0.5 else "❌"
        print(f"\n{overall_status} OVERALL INTEGRATION: {overall_score:.0%}")
        
        if overall_score >= 0.8:
            print(f"\n🎉 LOCAL LLM INTEGRATION SUCCESSFUL!")
            print(f"✅ Smart arbitration working with local vLLM server")
            print(f"✅ Low latency responses enable frequent LLM consultation")
            print(f"✅ Context-aware triggers optimally integrated")
            
            # Performance recommendations
            conn = all_results.get('connectivity', {})
            planner = all_results.get('planner_integration', {})
            arb = all_results.get('arbitration_integration', {})
            
            print(f"\n🚀 PERFORMANCE OPTIMIZATIONS ENABLED:")
            print(f"• Server response time: {conn.get('response_time_ms', 0):.1f}ms")
            print(f"• LLM inference latency: {planner.get('average_latency_ms', 0):.1f}ms") 
            print(f"• Smart arbitration calls: {arb.get('local_llm_calls', 0)}")
            print(f"• Context triggers working: {len(arb.get('triggers_detected', []))}/4")
            
            print(f"\n🎯 RECOMMENDED CONFIGURATION:")
            print(f"• Base arbitration frequency: 50-80 steps (was 150)")
            print(f"• Min frequency: 20 steps (was 50)")
            print(f"• Macro timeout: 40-60 steps (was 75)")
            print(f"• Enable fast mode and caching")
            print(f"• Use local_llm_planner.py instead of remote planner")
            
        else:
            print(f"\n⚠️ Integration needs optimization (score: {overall_score:.0%})")
            print(f"Review failed components above")
        
        return overall_score >= 0.8
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
