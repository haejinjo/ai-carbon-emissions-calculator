#!/usr/bin/env python3
"""
Comprehensive accuracy tests for AI Carbon Emissions Calculator
Run this file to verify the calculator produces accurate results
"""

import json
from datetime import datetime

# Import the actual calculator
# Adjust the import based on your file structure:
try:
    # If the calculator is in a file called 'ai_carbon_calculator.py'
    from ai_emissions_calculator import AIEmissionsCalculator
except ImportError:
    try:
        # If it's in the same file or different name
        from carbon_calculator import AIEmissionsCalculator
    except ImportError:
        # If all else fails, define a minimal version here
        print("WARNING: Could not import AIEmissionsCalculator")
        print("Make sure your calculator file is in the same directory")
        print("and named 'ai_carbon_calculator.py' or 'carbon_calculator.py'")
        exit(1)


def test_manual_calculation():
    """Test 1: Verify basic math with manual calculation"""
    print("\n" + "="*60)
    print("TEST 1: Manual Calculation Verification")
    print("="*60)
    
    # Known values for manual check
    power = 100  # Watts
    time = 1     # hour
    carbon_intensity = 203.5  # gCO2/kWh (California)
    pue = 1.135  # AWS PUE
    
    # Manual calculation
    manual_energy = power / 1000  # kWh
    manual_emissions = manual_energy * carbon_intensity * pue / 1000
    
    print(f"Manual calculation:")
    print(f"  Power: {power}W × {time}h = {manual_energy} kWh")
    print(f"  Emissions: {manual_energy} × {carbon_intensity} × {pue} / 1000")
    print(f"  Result: {manual_emissions:.4f} kg CO2")
    
    # Calculator result (CPU is 150W with 0.65 utilization = 97.5W)
    calculator = AIEmissionsCalculator()
    result = calculator.estimate_emissions(
        model_size='small',
        tokens_processed=1000,
        instance_type='cpu',
        region='CA',
        provider='aws',
        duration_hours=1
    )
    
    print(f"\nCalculator result:")
    print(f"  Power used: {result['power_consumption_watts']}W")
    print(f"  Emissions: {result['total_emissions_kgCO2e']} kg CO2")
    
    # Check if results are close (within 20%)
    expected = (97.5 / 1000) * carbon_intensity * pue / 1000
    difference = abs(result['total_emissions_kgCO2e'] - expected) / expected * 100
    
    status = "✅ PASS" if difference < 20 else "❌ FAIL"
    print(f"\n{status} - Difference: {difference:.1f}%")
    
    return difference < 20


def test_cloud_provider_comparison():
    """Test 2: Compare with cloud provider estimates"""
    print("\n" + "="*60)
    print("TEST 2: Cloud Provider Comparison")
    print("="*60)
    
    calculator = AIEmissionsCalculator()
    
    # Test scenario: 1 month of CPU usage in different regions
    test_configs = [
        ('AWS Virginia', 'us-east-1', 'aws'),
        ('GCP Oregon', 'us-west1', 'gcp'),
        ('Azure California', 'us-west-1', 'azure'),
    ]
    
    print("1 CPU running for 1 month (720 hours):")
    print("-" * 40)
    
    results = []
    for name, region, provider in test_configs:
        result = calculator.estimate_emissions(
            model_size='small',
            tokens_processed=1,
            instance_type='cpu',
            region=region,
            provider=provider,
            duration_hours=720
        )
        
        results.append((name, result))
        print(f"{name:20} {result['total_emissions_kgCO2e']:8.2f} kg CO2")
        print(f"{'':20} Energy: {result['energy_consumption_kWh']:.1f} kWh")
        print(f"{'':20} Carbon: {result['carbon_intensity_gCO2_per_kWh']:.1f} gCO2/kWh")
        print()
    
    # Verify results are in reasonable range (10-100 kg for a month)
    all_in_range = all(10 < r[1]['total_emissions_kgCO2e'] < 100 for r in results)
    status = "✅ PASS" if all_in_range else "❌ FAIL"
    print(f"{status} - All results in expected range (10-100 kg/month)")
    
    return all_in_range

# Simulates training jobs for GPT-3, BERT, BLOOM, and compares emissions against published papers
def test_published_benchmarks():
    """Test 3: Compare against published ML training benchmarks"""
    print("\n" + "="*60)
    print("TEST 3: Published Benchmark Comparison")
    print("="*60)
    
    calculator = AIEmissionsCalculator()
    
    # GPT-3 benchmark: ~552,000 kg CO2 for training
    # Estimate: 1000 V100 GPUs for 30 days
    print("Simulating GPT-3 training (1000 V100s for 30 days):")
    print("-" * 40)
    
    # Single V100 for 720 hours
    result = calculator.estimate_emissions(
        model_size='xlarge',
        tokens_processed=1,
        instance_type='v100',
        region='US_AVG',
        provider='azure',
        duration_hours=720,
        task_type='training'
    )
    
    # Scale to 1000 GPUs
    single_gpu_emissions = result['total_emissions_kgCO2e']
    total_emissions = single_gpu_emissions * 1000
    
    print(f"Single V100 for 1 month: {single_gpu_emissions:.1f} kg CO2")
    print(f"1000 V100s for 1 month: {total_emissions:,.0f} kg CO2")
    print(f"GPT-3 reported: 552,000 kg CO2")
    print(f"Ratio: {total_emissions / 552000:.2f}x")
    
    # Check if within reasonable range (0.2x to 5x)
    ratio = total_emissions / 552000
    in_range = 0.2 < ratio < 5
    status = "✅ PASS" if in_range else "❌ FAIL"
    print(f"\n{status} - Within expected range (0.2x - 5x)")
    
    # Additional benchmarks
    print("\nOther model benchmarks:")
    benchmarks = [
        ('BERT training', 'medium', 'v100', 24, 22.7),
        ('GPT-2 training', 'medium', 'v100', 168, 150),
    ]
    
    for name, size, gpu, hours, expected in benchmarks:
        result = calculator.estimate_emissions(
            model_size=size,
            tokens_processed=1_000_000,
            instance_type=gpu,
            region='US_AVG',
            provider='aws',
            duration_hours=hours,
            task_type='training'
        )
        
        ratio = result['total_emissions_kgCO2e'] / expected
        print(f"{name}: {result['total_emissions_kgCO2e']:.1f} kg (expected ~{expected} kg, ratio: {ratio:.2f})")
    
    return in_range

# Manually computes expected kWh for given instance types (e.g. A100, T4, H100) using raw wattage × time × utilization
# Compares to emissions calculator output
# Asserts percent difference and flags if beyond threshold:
def test_energy_calculation():
    """Test 4: Verify energy calculations"""
    print("\n" + "="*60)
    print("TEST 4: Energy Calculation Verification")
    print("="*60)
    
    calculator = AIEmissionsCalculator()
    
    # Test with known energy consumption
    test_cases = [
        ("A100 at full power for 1 hour", 'a100', 1, 400 * 0.65 / 1000),
        ("T4 inference for 10 hours", 't4', 10, 70 * 0.65 * 10 / 1000),
        ("H100 training for 1 hour", 'h100', 1, 700 * 0.90 * 1.5 / 1000),
    ]
    
    all_pass = True
    for name, gpu, hours, expected_kwh in test_cases:
        is_training = 'training' in name
        
        result = calculator.estimate_emissions(
            model_size='large',
            tokens_processed=1000,
            instance_type=gpu,
            region='US_AVG',
            provider='aws',
            duration_hours=hours,
            task_type='training' if is_training else 'inference'
        )
        
        # For training, expected kWh doesn't include the 1.5x factor in energy
        if is_training:
            expected_kwh = expected_kwh / 1.5
        
        difference = abs(result['energy_consumption_kWh'] - expected_kwh) / expected_kwh * 100
        status = "✅" if difference < 5 else "❌"
        
        print(f"{name}:")
        print(f"  Expected: {expected_kwh:.3f} kWh")
        print(f"  Calculated: {result['energy_consumption_kWh']:.3f} kWh")
        print(f"  {status} Difference: {difference:.1f}%")
        
        if difference >= 5:
            all_pass = False
    
    overall_status = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"\n{overall_status} - Energy calculations")
    
    return all_pass


def test_sanity_ranges():
    """Test 5: Sanity check emission ranges"""
    print("\n" + "="*60)
    print("TEST 5: Sanity Check - Emission Ranges")
    print("="*60)
    
    calculator = AIEmissionsCalculator()
    
    test_scenarios = [
        ("1 hour inference", 'inference', 1, 0.001, 0.5),
        ("8 hour workday", 'inference', 8, 0.01, 4),
        ("1 day training", 'training', 24, 1, 50),
        ("1 week training", 'training', 168, 10, 500),
        ("1 month training", 'training', 720, 50, 2000),
    ]
    
    print(f"{'Scenario':<20} {'Emissions':<15} {'Expected Range':<20} {'Status'}")
    print("-" * 70)
    
    all_pass = True
    for name, task, hours, min_kg, max_kg in test_scenarios:
        result = calculator.estimate_emissions(
            model_size='large',
            tokens_processed=1_000_000,
            instance_type='a100',
            region='US_AVG',
            provider='aws',
            task_type=task,
            duration_hours=hours
        )
        
        emissions = result['total_emissions_kgCO2e']
        in_range = min_kg <= emissions <= max_kg
        status = "✅ PASS" if in_range else "❌ FAIL"
        
        print(f"{name:<20} {emissions:>10.3f} kg   {min_kg:>6.1f} - {max_kg:<8.1f} kg   {status}")
        
        if not in_range:
            all_pass = False
    
    return all_pass

# Runs the same workload in regions with different carbon intensities (e.g. NY = 129.9, TX = 396.0, MO = 675.5)
# Confirms whether higher emissions correlate with dirtier grid
# Validates ratio between emissions and carbon intensity:
def test_regional_variations():
    """Test 6: Verify regional carbon intensity differences"""
    print("\n" + "="*60)
    print("TEST 6: Regional Carbon Intensity Variations")
    print("="*60)
    
    calculator = AIEmissionsCalculator()
    
    # Test same workload in different regions
    regions = [
        ('New York (cleanest)', 'NY', 129.9),
        ('California', 'CA', 203.5),
        ('Texas', 'TX', 396.0),
        ('Missouri (dirtiest)', 'SRMW', 675.5),
    ]
    
    print("Same workload (1M tokens) in different regions:")
    print("-" * 50)
    
    base_result = None
    all_pass = True
    
    for name, region, expected_intensity in regions:
        result = calculator.estimate_emissions(
            model_size='large',
            tokens_processed=1_000_000,
            instance_type='a100',
            region=region,
            provider='aws',
            duration_hours=1
        )
        
        # Verify carbon intensity
        intensity_match = abs(result['carbon_intensity_gCO2_per_kWh'] - expected_intensity) < 1
        status = "✅" if intensity_match else "❌"
        
        if not base_result:
            base_result = result
            ratio = 1.0
        else:
            ratio = result['total_emissions_kgCO2e'] / base_result['total_emissions_kgCO2e']
        
        print(f"{name:<25} {result['total_emissions_kgCO2e']:>8.4f} kg CO2  "
              f"(intensity: {result['carbon_intensity_gCO2_per_kWh']:>5.1f}, "
              f"ratio: {ratio:>4.1f}x) {status}")
        
        if not intensity_match:
            all_pass = False
    
    # Verify Missouri is ~5x dirtier than New York
    ny_result = calculator.estimate_emissions(
        model_size='large', tokens_processed=1_000_000,
        instance_type='a100', region='NY', provider='aws', duration_hours=1
    )
    mo_result = calculator.estimate_emissions(
        model_size='large', tokens_processed=1_000_000,
        instance_type='a100', region='SRMW', provider='aws', duration_hours=1
    )
    
    ratio = mo_result['total_emissions_kgCO2e'] / ny_result['total_emissions_kgCO2e']
    expected_ratio = 675.5 / 129.9
    ratio_correct = abs(ratio - expected_ratio) / expected_ratio < 0.1
    
    print(f"\nMissouri vs New York ratio: {ratio:.1f}x (expected {expected_ratio:.1f}x)")
    final_status = "✅ PASS" if all_pass and ratio_correct else "❌ FAIL"
    print(f"{final_status} - Regional variations")
    
    return all_pass and ratio_correct


def run_all_tests():
    """Run all accuracy tests"""
    print("\n" + "="*60)
    print("AI CARBON CALCULATOR - ACCURACY TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Manual Calculation", test_manual_calculation),
        ("Cloud Provider Comparison", test_cloud_provider_comparison),
        ("Published Benchmarks", test_published_benchmarks),
        ("Energy Calculation", test_energy_calculation),
        ("Sanity Ranges", test_sanity_ranges),
        ("Regional Variations", test_regional_variations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("ACCURACY TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All accuracy tests passed! The calculator is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Review the results above.")
    
    return passed == total


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)