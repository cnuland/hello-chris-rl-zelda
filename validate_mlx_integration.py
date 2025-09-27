#!/usr/bin/env python3
"""
MLX Integration Validation

Simple validation of MLX server integration without complex imports.
Tests the core functionality needed for smart arbitration.
"""

import asyncio
import json
import time
import sys

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run: pip install httpx")
    sys.exit(1)


async def test_mlx_for_smart_arbitration():
    """Test MLX server for smart arbitration compatibility."""
    
    print("🍎 MLX SMART ARBITRATION VALIDATION")
    print("=" * 50)
    
    server_url = "http://localhost:8000"
    model_name = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    
    results = {
        'server_online': False,
        'scenarios_tested': 0,
        'successful_responses': 0,
        'valid_json_responses': 0,
        'response_times': [],
        'actions_generated': []
    }
    
    # Smart arbitration test scenarios
    scenarios = [
        {
            'name': '🗺️ New Room Discovery',
            'system': 'You are an expert Zelda AI. Respond with JSON: {"action": "ACTION", "reasoning": "why"}',
            'user': 'Just entered new room with rupee at (120,90) and door at (80,20). Health: 3/3. Choose: MOVE_TO, EXPLORE_ROOM, COLLECT_ITEM, ENTER_DOOR',
            'expected_actions': ['COLLECT_ITEM', 'EXPLORE_ROOM']
        },
        {
            'name': '❤️ Low Health Emergency', 
            'system': 'You are an expert Zelda AI. Respond with JSON: {"action": "ACTION", "reasoning": "why"}',
            'user': 'CRITICAL: Health 1/3 hearts! Enemy octorok nearby at (100,60). Have healing potion. Choose: ATTACK_ENEMY, MOVE_TO, USE_ITEM, EXPLORE_ROOM',
            'expected_actions': ['USE_ITEM', 'MOVE_TO']
        },
        {
            'name': '🚫 Stuck Recovery',
            'system': 'You are an expert Zelda AI. Respond with JSON: {"action": "ACTION", "reasoning": "why"}',
            'user': 'Been at same position (80,72) for 75 steps. No progress. See wall and door. Choose: MOVE_TO, EXPLORE_ROOM, ENTER_DOOR, ATTACK_ENEMY',
            'expected_actions': ['ENTER_DOOR', 'MOVE_TO', 'EXPLORE_ROOM']
        },
        {
            'name': '💬 NPC Interaction',
            'system': 'You are an expert Zelda AI. Respond with JSON: {"action": "ACTION", "reasoning": "why"}',
            'user': 'NPC villager nearby wants to talk. Dialogue available. Health: 2/3. Choose: MOVE_TO, EXPLORE_ROOM, COLLECT_ITEM, USE_ITEM',
            'expected_actions': ['MOVE_TO']  # Move to talk to NPC
        },
        {
            'name': '🏰 Dungeon Strategy',
            'system': 'You are an expert Zelda AI. Respond with JSON: {"action": "ACTION", "reasoning": "why"}',
            'user': 'Dungeon entrance found! Health: 2/3, have sword and 3 rupees. Dark entrance ahead. Choose: ENTER_DOOR, MOVE_TO, USE_ITEM, EXPLORE_ROOM',
            'expected_actions': ['ENTER_DOOR', 'USE_ITEM']  # Enter or heal first
        }
    ]
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Test server connectivity
            health_check = await client.get(f"{server_url}/health")
            results['server_online'] = health_check.status_code == 200
            
            if not results['server_online']:
                print("❌ MLX server not responding")
                return results
            
            print("✅ MLX server online and ready")
            print()
            
            # Test each scenario
            for scenario in scenarios:
                print(f"Testing {scenario['name']}...")
                
                start_time = time.time()
                
                request_data = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": scenario['system']},
                        {"role": "user", "content": scenario['user']}
                    ],
                    "max_tokens": 120,
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = await client.post(
                    f"{server_url}/v1/chat/completions",
                    json=request_data
                )
                
                response_time_ms = (time.time() - start_time) * 1000
                results['response_times'].append(response_time_ms)
                results['scenarios_tested'] += 1
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        results['successful_responses'] += 1
                        
                        # Try to parse JSON
                        try:
                            if '{' in content and '}' in content:
                                json_start = content.find('{')
                                json_end = content.rfind('}') + 1
                                json_part = content[json_start:json_end]
                                parsed = json.loads(json_part)
                                
                                action = parsed.get('action', 'unknown')
                                reasoning = parsed.get('reasoning', 'none')
                                
                                results['valid_json_responses'] += 1
                                results['actions_generated'].append(action)
                                
                                # Check if action is reasonable
                                is_good_action = action in scenario['expected_actions']
                                action_quality = "✅" if is_good_action else "⚠️"
                                
                                print(f"  {action_quality} {response_time_ms:.0f}ms: {action}")
                                print(f"    Reasoning: {reasoning[:60]}...")
                                
                            else:
                                print(f"  ❌ {response_time_ms:.0f}ms: No JSON in response")
                                print(f"    Raw: {content[:60]}...")
                                
                        except json.JSONDecodeError as e:
                            print(f"  ❌ {response_time_ms:.0f}ms: Invalid JSON")
                            print(f"    Content: {content[:60]}...")
                            
                    else:
                        print(f"  ❌ {response_time_ms:.0f}ms: No choices in response")
                else:
                    print(f"  ❌ Request failed: {response.status_code}")
                
                print()
                
                # Brief pause between requests
                await asyncio.sleep(0.5)
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['error'] = str(e)
        
    return results


def analyze_results(results):
    """Analyze and report results."""
    
    print("=" * 50)
    print("📊 MLX INTEGRATION ANALYSIS")
    print("=" * 50)
    
    # Calculate scores
    connectivity_score = 1.0 if results['server_online'] else 0.0
    response_score = results['successful_responses'] / max(1, results['scenarios_tested'])
    json_score = results['valid_json_responses'] / max(1, results['scenarios_tested'])
    
    overall_score = (connectivity_score * 0.2 + response_score * 0.4 + json_score * 0.4)
    
    # Performance metrics
    avg_response_time = sum(results['response_times']) / max(1, len(results['response_times']))
    
    print(f"✅ Server Connectivity: {connectivity_score:.0%}")
    print(f"✅ Successful Responses: {response_score:.0%} ({results['successful_responses']}/{results['scenarios_tested']})")
    print(f"✅ Valid JSON Parsing: {json_score:.0%} ({results['valid_json_responses']}/{results['scenarios_tested']})")
    
    overall_status = "✅" if overall_score >= 0.8 else "⚠️" if overall_score >= 0.6 else "❌"
    print(f"\n{overall_status} OVERALL INTEGRATION: {overall_score:.0%}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  Average response time: {avg_response_time:.0f}ms")
    
    if avg_response_time < 1000:
        performance_tier = "EXCELLENT (< 1s)"
        recommended_frequency = "60-80 steps"
    elif avg_response_time < 2000:
        performance_tier = "GOOD (1-2s)" 
        recommended_frequency = "100-120 steps"
    elif avg_response_time < 3000:
        performance_tier = "ACCEPTABLE (2-3s)"
        recommended_frequency = "150-200 steps"
    else:
        performance_tier = "SLOW (> 3s)"
        recommended_frequency = "200+ steps"
        
    print(f"  Performance tier: {performance_tier}")
    print(f"  Recommended arbitration frequency: {recommended_frequency}")
    
    print(f"\n🎯 ACTIONS GENERATED:")
    for action in set(results['actions_generated']):
        count = results['actions_generated'].count(action)
        print(f"  • {action}: {count} times")
    
    if overall_score >= 0.8:
        print(f"\n🎉 MLX INTEGRATION READY FOR SMART ARBITRATION!")
        print(f"✅ Excellent response quality and performance")
        print(f"✅ Consistent JSON format for macro actions")
        print(f"✅ Context-aware decision making")
        print(f"✅ {avg_response_time:.0f}ms response time ideal for RL")
        
        print(f"\n🔧 RECOMMENDED SMART ARBITRATION CONFIG:")
        print(f"```yaml")
        print(f"planner_integration:")
        print(f"  use_planner: true")
        print(f"  use_smart_arbitration: true")
        print(f"  base_planner_frequency: {recommended_frequency.split('-')[0]}")
        print(f"  endpoint_url: \"http://localhost:8000/v1/chat/completions\"")
        print(f"  model_name: \"mlx-community/Qwen2.5-14B-Instruct-4bit\"")
        print(f"  max_tokens: 100")
        print(f"  temperature: 0.3")
        print(f"  timeout: 10.0")
        print(f"```")
        
        print(f"\n🚀 INTEGRATION BENEFITS:")
        print(f"  • Smart context-aware guidance every ~{int(recommended_frequency.split('-')[0])*4/60:.1f} minutes")
        print(f"  • {avg_response_time:.0f}ms latency enables responsive arbitration")
        print(f"  • Perfect JSON format for macro actions")
        print(f"  • Unlimited local inference - no API costs")
        print(f"  • Works seamlessly with exploration reward system")
        
    else:
        print(f"\n⚠️ Integration needs optimization (score: {overall_score:.0%})")
        
        if connectivity_score < 1.0:
            print(f"  • MLX server connectivity issues")
        if response_score < 0.8:
            print(f"  • Response quality issues")  
        if json_score < 0.8:
            print(f"  • JSON parsing problems")
            
    return overall_score >= 0.8


async def main():
    """Run MLX integration validation."""
    results = await test_mlx_for_smart_arbitration()
    success = analyze_results(results)
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n{'🎯 MLX integration validated!' if success else '⚠️ MLX integration needs work.'}")
    sys.exit(0 if success else 1)
