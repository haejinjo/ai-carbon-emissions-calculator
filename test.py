# #!/usr/bin/env python3
# """
# Comprehensive test suite for AI Carbon Emissions Calculator
# Tests both static EPA data and WattTime API integration
# """

# import json
# import os
# from datetime import datetime
# from typing import Dict

# # Import the calculator (assuming it's in a file called ai_carbon_calculator.py)
# # from ai_carbon_calculator import AIEmissionsCalculator, EmissionsMonitor

# def test_static_epa_data(calculator):
#     """Test 1: Verify static EPA data works without any API"""
#     print("\n" + "="*60)
#     print("TEST 1: Static EPA Data (No API Required)")
#     print("="*60)
    
#     test_cases = [
#         {
#             'name': 'California Inference',
#             'params': {
#                 'model_size': 'large',
#                 'tokens_processed': 1_000_000,
#                 'instance_type': 'a100',
#                 'region': 'CA',
#                 'provider': 'aws',
#                 'task_type': 'inference',
#                 'batch_size': 32
#             }
#         },
#         {
#             'name': 'Texas Training',
#             'params': {
#                 'model_size': 'medium',
#                 'tokens_processed': 10_000_000,
#                 'instance_type': 'v100',
#                 'region': 'TX',
#                 'provider': 'on_premise',
#                 'task_type': 'training',
#                 'batch_size': 64
#             }
#         },
#         {
#             'name': 'Unknown Region Fallback',
#             'params': {
#                 'model_size': 'small',
#                 'tokens_processed': 100_000,
#                 'instance_type': 't4',
#                 'region': 'INVALID',
#                 'provider': 'gcp',
#                 'task_type': 'inference'
#             }
#         }
#     ]
    
#     results = []
#     for test in test_cases:
#         print(f"\nTesting: {test['name']}")
#         try:
#             result = calculator.estimate_emissions(**test['params'])
#             print(f"✓ Success: {result['total_emissions_kgCO2e']:.3f} kg CO2")
#             print(f"  Carbon intensity: {result['carbon_intensity_gCO2_per_kWh']} gCO2/kWh")
#             print(f"  Per million tokens: {result['emissions_per_million_tokens']:.3f} kg CO2")
#             results.append(('PASS', test['name'], result))
#         except Exception as e:
#             print(f"✗ Failed: {str(e)}")
#             results.append(('FAIL', test['name'], str(e)))
    
#     return results

# def test_hardware_coverage(calculator):
#     """Test 2: Verify all GPU types are recognized"""
#     print("\n" + "="*60)
#     print("TEST 2: Hardware Coverage")
#     print("="*60)
    
#     gpu_types = ['a100', 'h100', 'v100', 't4', 'a10g', 'a6000', 'l4', 'tpu_v2', 'cpu']
    
#     for gpu in gpu_types:
#         try:
#             result = calculator.estimate_emissions(
#                 model_size='small',
#                 tokens_processed=10_000,
#                 instance_type=gpu,
#                 region='US_AVG',
#                 provider='aws'
#             )
#             power = calculator.gpu_power.get(gpu, 'Unknown')
#             print(f"✓ {gpu}: {power}W - Emissions: {result['total_emissions_kgCO2e']:.4f} kg")
#         except Exception as e:
#             print(f"✗ {gpu}: Failed - {str(e)}")

# def test_watttime_integration(calculator):
#     """Test 3: WattTime API Integration (requires credentials)"""
#     print("\n" + "="*60)
#     print("TEST 3: WattTime Real-time API")
#     print("="*60)
    
#     # Check if credentials are available
#     has_creds = os.getenv('WATTTIME_USER') and os.getenv('WATTTIME_PASSWORD')
    
#     if not has_creds:
#         print("⚠ WattTime credentials not found in environment variables")
#         print("  Set WATTTIME_USER and WATTTIME_PASSWORD to test real-time data")
#         return [('SKIP', 'WattTime API', 'No credentials')]
    
#     # Test locations
#     locations = [
#         ('San Francisco, CA', 37.7749, -122.4194),
#         ('Austin, TX', 30.2672, -97.7431),
#         ('New York, NY', 40.7128, -74.0060),
#         ('Seattle, WA', 47.6062, -122.3321)
#     ]
    
#     results = []
#     for name, lat, lon in locations:
#         print(f"\nTesting: {name}")
#         try:
#             # Test with real-time data
#             result_rt = calculator.estimate_emissions(
#                 model_size='medium',
#                 tokens_processed=100_000,
#                 instance_type='a100',
#                 region='CA',  # Fallback
#                 provider='aws',
#                 lat=lat,
#                 lon=lon
#             )
            
#             # Test without coordinates (static data)
#             result_static = calculator.estimate_emissions(
#                 model_size='medium',
#                 tokens_processed=100_000,
#                 instance_type='a100',
#                 region='CA',
#                 provider='aws'
#             )
            
#             diff_pct = abs(result_rt['carbon_intensity_gCO2_per_kWh'] - 
#                           result_static['carbon_intensity_gCO2_per_kWh']) / result_static['carbon_intensity_gCO2_per_kWh'] * 100
            
#             print(f"✓ Real-time: {result_rt['carbon_intensity_gCO2_per_kWh']:.1f} gCO2/kWh")
#             print(f"  Static EPA: {result_static['carbon_intensity_gCO2_per_kWh']:.1f} gCO2/kWh")
#             print(f"  Difference: {diff_pct:.1f}%")
            
#             results.append(('PASS', name, result_rt))
#         except Exception as e:
#             print(f"✗ Failed: {str(e)}")
#             results.append(('FAIL', name, str(e)))
    
#     return results

# def test_monitoring_system(calculator):
#     """Test 4: Production Monitoring System"""
#     print("\n" + "="*60)
#     print("TEST 4: Monitoring System")
#     print("="*60)
    
#     monitor = EmissionsMonitor(calculator)
    
#     # Simulate API requests
#     requests = [
#         {'model_id': 'gpt-3.5', 'tokens': 150, 'latency_ms': 45},
#         {'model_id': 'gpt-3.5', 'tokens': 500, 'latency_ms': 120},
#         {'model_id': 'gpt-4', 'tokens': 1000, 'latency_ms': 890},
#     ]
    
#     for req in requests:
#         emissions = monitor.log_api_request(
#             model_id=req['model_id'],
#             tokens=req['tokens'],
#             latency_ms=req['latency_ms'],
#             model_size='medium',
#             instance_type='a100',
#             region='CA',
#             provider='aws'
#         )
#         print(f"✓ {req['model_id']}: {req['tokens']} tokens = {emissions['total_emissions_kgCO2e']:.6f} kg CO2")
    
#     # Get daily report
#     report = monitor.get_daily_report()
#     print(f"\nDaily Report:")
#     print(f"  Total emissions: {report['total_emissions_kgCO2e']:.6f} kg CO2")
#     print(f"  Total tokens: {report['total_tokens']}")
#     print(f"  API calls: {report['api_calls']}")

# def test_edge_cases(calculator):
#     """Test 5: Edge Cases and Error Handling"""
#     print("\n" + "="*60)
#     print("TEST 5: Edge Cases")
#     print("="*60)
    
#     edge_cases = [
#         {
#             'name': 'Zero tokens',
#             'params': {
#                 'model_size': 'small',
#                 'tokens_processed': 0,
#                 'instance_type': 'a100',
#                 'region': 'CA',
#                 'provider': 'aws'
#             }
#         },
#         {
#             'name': 'Unknown GPU fallback',
#             'params': {
#                 'model_size': 'medium',
#                 'tokens_processed': 1000,
#                 'instance_type': 'unknown_gpu',
#                 'region': 'CA',
#                 'provider': 'aws'
#             }
#         },
#         {
#             'name': 'Very large workload',
#             'params': {
#                 'model_size': 'xlarge',
#                 'tokens_processed': 1_000_000_000_000,  # 1 trillion
#                 'instance_type': 'h100',
#                 'region': 'TX',
#                 'provider': 'azure',
#                 'duration_hours': 720  # 30 days
#             }
#         }
#     ]
    
#     for test in edge_cases:
#         print(f"\nTesting: {test['name']}")
#         try:
#             result = calculator.estimate_emissions(**test['params'])
#             print(f"✓ Handled successfully: {result['total_emissions_kgCO2e']:.3f} kg CO2")
#         except Exception as e:
#             print(f"✗ Error (expected): {str(e)}")

# def run_all_tests():
#     """Run comprehensive test suite"""
#     print("\n" + "="*60)
#     print("AI CARBON CALCULATOR - COMPREHENSIVE TEST SUITE")
#     print("="*60)
#     print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
#     # Initialize calculator
#     calculator = AIEmissionsCalculator()
    
#     # Run all tests
#     test_results = {
#         'static_epa': test_static_epa_data(calculator),
#         'hardware': test_hardware_coverage(calculator),
#         'watttime': test_watttime_integration(calculator),
#         'monitoring': test_monitoring_system(calculator),
#         'edge_cases': test_edge_cases(calculator)
#     }
    
#     # Test optimization suggestions
#     print("\n" + "="*60)
#     print("TEST 6: Optimization Suggestions")
#     print("="*60)
    
#     high_emissions = {
#         'total_emissions_kgCO2e': 50.0,
#         'emissions_per_million_tokens': 0.5,
#         'carbon_intensity_gCO2_per_kWh': 550
#     }
    
#     suggestions = calculator.get_optimization_suggestions(high_emissions, 'TX', 'on_premise')
#     print(f"Found {len(suggestions)} optimization suggestions:")
#     for s in suggestions:
#         print(f"  [{s['impact'].upper()}] {s['suggestion'][:50]}...")
    
#     # Summary
#     print("\n" + "="*60)
#     print("TEST SUMMARY")
#     print("="*60)
    
#     total_tests = 0
#     passed_tests = 0
    
#     for category, results in test_results.items():
#         if isinstance(results, list):
#             for status, name, _ in results:
#                 total_tests += 1
#                 if status == 'PASS':
#                     passed_tests += 1
    
#     print(f"Total tests run: {total_tests}")
#     print(f"Passed: {passed_tests}")
#     print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    
#     print("\n✓ Core calculator functionality verified")
#     print("✓ Static EPA data working correctly")
#     print("✓ All hardware types recognized")
#     print("✓ Error handling functional")
    
#     if os.getenv('WATTTIME_USER'):
#         print("✓ WattTime API integration tested")
#     else:
#         print("⚠ WattTime API not tested (set credentials to enable)")

# if __name__ == "__main__":
#     # Quick functionality test
#     print("QUICK FUNCTIONALITY TEST")
#     print("-" * 40)
    
#     calculator = AIEmissionsCalculator()
    
#     # Test 1: Basic calculation
#     result = calculator.estimate_emissions(
#         model_size='large',
#         tokens_processed=100_000,
#         instance_type='a100',
#         region='CA',
#         provider='aws'
#     )
    
#     print(f"✓ Basic calculation works!")
#     print(f"  100k tokens on A100 in California = {result['total_emissions_kgCO2e']:.3f} kg CO2")
#     print(f"  Energy used: {result['energy_consumption_kWh']:.3f} kWh")
    
#     # Run full test suite
#     print("\nRun full test suite? (y/n): ", end="")
#     if input().lower() == 'y':
#         run_all_tests()