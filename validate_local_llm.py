#!/usr/bin/env python3
"""
Local LLM Validation Script

Tests connection to the local vLLM server and validates it's working
properly for smart arbitration integration.
"""

import asyncio
import json
import time
import sys
from typing import Dict, Any

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run: pip install httpx")
    sys.exit(1)


class LocalLLMValidator:
    """Validates local vLLM server for smart arbitration."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        
    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic connection to local vLLM server."""
        results = {
            'server_reachable': False,
            'health_endpoint': False,
            'response_time_ms': 0,
            'error': None
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                print(f"🔗 Testing connection to {self.server_url}...")
                
                start_time = time.time()
                response = await client.get(f"{self.server_url}/health")
                response_time = (time.time() - start_time) * 1000
                
                results['server_reachable'] = True
                results['response_time_ms'] = response_time
                results['health_endpoint'] = response.status_code == 200
                
                print(f"  ✅ Server reachable: {response.status_code}")
                print(f"  ⚡ Response time: {response_time:.1f}ms")
                
        except Exception as e:
            results['error'] = str(e)
            print(f"  ❌ Connection failed: {e}")
            
        return results
    
    async def test_generation_endpoint(self) -> Dict[str, Any]:
        """Test the /generate endpoint with a simple request."""
        results = {
            'endpoint_working': False,
            'response_time_ms': 0,
            'response_received': False,
            'response_content': '',
            'error': None
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                print(f"\n🧠 Testing /generate endpoint...")
                
                # Simple test request
                test_request = {
                    "prompt": "You are a helpful AI assistant. Respond with a simple JSON: {\"action\": \"test\", \"status\": \"working\"}",
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stream": False,
                    "stop": ["\n\n"]
                }
                
                start_time = time.time()
                response = await client.post(
                    f"{self.server_url}/generate",
                    json=test_request,
                    timeout=10.0
                )
                response_time = (time.time() - start_time) * 1000
                
                results['response_time_ms'] = response_time
                results['endpoint_working'] = response.status_code == 200
                
                if response.status_code == 200:
                    response_json = response.json()
                    results['response_received'] = True
                    
                    # Extract generated text
                    generated_text = ""
                    if 'text' in response_json:
                        generated_text = response_json['text'][0] if isinstance(response_json['text'], list) else response_json['text']
                    elif 'choices' in response_json and len(response_json['choices']) > 0:
                        generated_text = response_json['choices'][0].get('text', '')
                    
                    results['response_content'] = generated_text.strip()
                    
                    print(f"  ✅ Generation working: {response_time:.1f}ms")
                    print(f"  📝 Response: {generated_text.strip()[:100]}...")
                    
                else:
                    print(f"  ❌ Generation failed: {response.status_code}")
                    print(f"  📝 Error: {response.text}")
                    
        except Exception as e:
            results['error'] = str(e)
            print(f"  ❌ Generation test failed: {e}")
            
        return results
    
    async def test_zelda_scenario(self) -> Dict[str, Any]:
        """Test with a Zelda-specific scenario."""
        results = {
            'scenario_working': False,
            'response_time_ms': 0,
            'valid_json_response': False,
            'response_content': '',
            'error': None
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                print(f"\n🎮 Testing Zelda scenario...")
                
                # Zelda-specific test
                zelda_prompt = """You are an expert AI playing The Legend of Zelda: Oracle of Seasons. 

Current game state:
- Health: 2/3 hearts
- Room: 5 
- Position: (120, 80)
- Enemies nearby: 1 octorok at (100, 60)
- Items nearby: rupee at (140, 100)

Choose ONE action and respond with JSON only:
{"action": "ATTACK_ENEMY", "target": "octorok", "priority": 3, "reasoning": "enemy is close and threatening"}

Actions: MOVE_TO, EXPLORE_ROOM, ATTACK_ENEMY, COLLECT_ITEM, USE_ITEM, ENTER_DOOR"""

                test_request = {
                    "prompt": zelda_prompt,
                    "max_tokens": 150,
                    "temperature": 0.3,
                    "stream": False,
                    "stop": ["\n\n", "Current game state:", "Actions:"]
                }
                
                start_time = time.time()
                response = await client.post(
                    f"{self.server_url}/generate",
                    json=test_request,
                    timeout=10.0
                )
                response_time = (time.time() - start_time) * 1000
                
                results['response_time_ms'] = response_time
                results['scenario_working'] = response.status_code == 200
                
                if response.status_code == 200:
                    response_json = response.json()
                    
                    # Extract generated text
                    generated_text = ""
                    if 'text' in response_json:
                        generated_text = response_json['text'][0] if isinstance(response_json['text'], list) else response_json['text']
                    elif 'choices' in response_json and len(response_json['choices']) > 0:
                        generated_text = response_json['choices'][0].get('text', '')
                    
                    results['response_content'] = generated_text.strip()
                    
                    # Check if response contains valid JSON
                    try:
                        if '{' in generated_text and '}' in generated_text:
                            json_start = generated_text.find('{')
                            json_end = generated_text.rfind('}') + 1
                            json_part = generated_text[json_start:json_end]
                            parsed = json.loads(json_part)
                            results['valid_json_response'] = True
                            results['parsed_action'] = parsed
                            
                            print(f"  ✅ Zelda scenario working: {response_time:.1f}ms")
                            print(f"  🎯 Action: {parsed.get('action', 'unknown')}")
                            print(f"  🎯 Target: {parsed.get('target', 'unknown')}")
                            print(f"  🎯 Reasoning: {parsed.get('reasoning', 'unknown')[:50]}...")
                        else:
                            print(f"  ⚠️ Response not JSON format: {generated_text[:100]}...")
                            
                    except json.JSONDecodeError:
                        print(f"  ⚠️ Invalid JSON in response: {generated_text[:100]}...")
                        
                else:
                    print(f"  ❌ Zelda scenario failed: {response.status_code}")
                    
        except Exception as e:
            results['error'] = str(e)
            print(f"  ❌ Zelda scenario test failed: {e}")
            
        return results
    
    def analyze_performance(self, connectivity: Dict, generation: Dict, scenario: Dict) -> Dict[str, Any]:
        """Analyze performance and provide recommendations."""
        
        # Calculate scores
        connectivity_score = 1.0 if connectivity.get('server_reachable') and connectivity.get('health_endpoint') else 0.0
        generation_score = 1.0 if generation.get('endpoint_working') and generation.get('response_received') else 0.0
        scenario_score = 1.0 if scenario.get('scenario_working') and scenario.get('valid_json_response') else 0.5 if scenario.get('scenario_working') else 0.0
        
        overall_score = (connectivity_score + generation_score + scenario_score) / 3.0
        
        # Latency analysis
        response_times = []
        if connectivity.get('response_time_ms'):
            response_times.append(connectivity['response_time_ms'])
        if generation.get('response_time_ms'):
            response_times.append(generation['response_time_ms'])
        if scenario.get('response_time_ms'):
            response_times.append(scenario['response_time_ms'])
            
        avg_latency = sum(response_times) / len(response_times) if response_times else 0
        
        # Performance tier
        if avg_latency < 200:
            performance_tier = "EXCELLENT (< 200ms)"
            recommended_frequency = "50-80 steps"
        elif avg_latency < 500:
            performance_tier = "GOOD (200-500ms)"
            recommended_frequency = "100-150 steps"
        elif avg_latency < 1000:
            performance_tier = "ACCEPTABLE (500-1000ms)"
            recommended_frequency = "150-200 steps"
        else:
            performance_tier = "SLOW (> 1000ms)"
            recommended_frequency = "200+ steps"
        
        return {
            'overall_score': overall_score,
            'connectivity_score': connectivity_score,
            'generation_score': generation_score,
            'scenario_score': scenario_score,
            'avg_latency_ms': avg_latency,
            'performance_tier': performance_tier,
            'recommended_frequency': recommended_frequency,
            'integration_ready': overall_score >= 0.8
        }


async def main():
    """Run the local LLM validation."""
    print("🚀 LOCAL LLM VALIDATION FOR SMART ARBITRATION")
    print("=" * 60)
    print("Testing local vLLM server for Zelda RL integration")
    print()
    
    validator = LocalLLMValidator()
    
    # Run all tests
    connectivity = await validator.test_basic_connectivity()
    
    if not connectivity.get('server_reachable'):
        print(f"\n❌ Cannot proceed - server not reachable")
        print(f"Make sure vLLM is running on http://localhost:8000")
        print(f"\nYour server logs show it should be running. Check:")
        print(f"  • Is the server still running?")
        print(f"  • Are there any firewall restrictions?")
        print(f"  • Try: curl http://localhost:8000/health")
        return False
    
    generation = await validator.test_generation_endpoint()
    scenario = await validator.test_zelda_scenario()
    
    # Analyze results
    analysis = validator.analyze_performance(connectivity, generation, scenario)
    
    # Print results
    print(f"\n" + "="*60)
    print("📊 LOCAL LLM VALIDATION RESULTS")
    print("="*60)
    
    print(f"✅ Connectivity: {analysis['connectivity_score']:.0%}")
    print(f"✅ Generation: {analysis['generation_score']:.0%}")
    print(f"✅ Zelda Scenario: {analysis['scenario_score']:.0%}")
    
    overall_status = "✅" if analysis['overall_score'] >= 0.8 else "⚠️" if analysis['overall_score'] >= 0.5 else "❌"
    print(f"\n{overall_status} OVERALL SCORE: {analysis['overall_score']:.0%}")
    
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print(f"  Average latency: {analysis['avg_latency_ms']:.1f}ms")
    print(f"  Performance tier: {analysis['performance_tier']}")
    print(f"  Recommended frequency: {analysis['recommended_frequency']}")
    
    if analysis['integration_ready']:
        print(f"\n🎉 LOCAL LLM READY FOR SMART ARBITRATION!")
        print(f"✅ Server responding correctly")
        print(f"✅ Generation endpoint working")
        print(f"✅ Zelda-specific scenarios functional")
        
        print(f"\n🔧 RECOMMENDED SMART ARBITRATION CONFIG:")
        print(f"```yaml")
        print(f"planner_integration:")
        print(f"  use_planner: true")
        print(f"  use_smart_arbitration: true")
        print(f"  ")
        print(f"  # OPTIMIZED FOR LOCAL LLM")
        print(f"  base_planner_frequency: {analysis['recommended_frequency'].split('-')[0]}  # Faster due to low latency")
        print(f"  min_planner_frequency: 20         # Can call more frequently")
        print(f"  max_planner_frequency: 150        # Less conservative limits")
        print(f"  macro_timeout: 50                 # Faster timeout")
        print(f"  ")
        print(f"  # LOCAL LLM ENDPOINT")
        print(f"  endpoint_url: \"http://localhost:8000/generate\"")
        print(f"  max_tokens: 128                   # Smaller for speed")
        print(f"  temperature: 0.3                  # More deterministic")
        print(f"  timeout: 5.0                      # Fast timeout for local")
        print(f"```")
        
        print(f"\n🚀 PERFORMANCE BENEFITS:")
        print(f"  • {analysis['avg_latency_ms']:.0f}ms average response time (vs ~2000ms for remote)")
        print(f"  • No API costs - unlimited requests")
        print(f"  • High availability - no network dependencies")
        print(f"  • Can call LLM {1000//analysis['avg_latency_ms']:.0f}x per second")
        print(f"  • Perfect for smart arbitration context triggers")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Update agents/planner.py to use localhost:8000")
        print(f"  2. Reduce arbitration frequencies in config")
        print(f"  3. Enable fast mode and caching")
        print(f"  4. Test with actual RL training")
        
    else:
        print(f"\n⚠️ Local LLM needs optimization for smart arbitration")
        print(f"Current score: {analysis['overall_score']:.0%} (target: 80%+)")
        
        if analysis['connectivity_score'] < 1.0:
            print(f"  • Fix connectivity issues first")
        if analysis['generation_score'] < 1.0:
            print(f"  • Generation endpoint not working properly")
        if analysis['scenario_score'] < 1.0:
            print(f"  • Zelda scenario responses need improvement")
    
    return analysis['integration_ready']


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
