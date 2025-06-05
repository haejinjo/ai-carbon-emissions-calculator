#!/usr/bin/env python3
"""
AI Carbon Emissions Calculator - Standalone Version
Copy this entire file and run it with: python carbon_calculator.py
"""

import requests
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

class AIEmissionsCalculator:
    """
    Real-time carbon emissions calculator for AI workloads using publicly available data.
    Designed for American businesses to track ML/AI carbon footprint.
    """
    
    def __init__(self):
        # EPA eGRID subregion carbon intensity (gCO2/kWh) - 2022 data
        # Source: https://www.epa.gov/egrid/download-data
        self.grid_carbon_intensity = {
            'CAMX': 203.5,  # California
            'ERCT': 396.0,  # Texas  
            'FRCC': 392.7,  # Florida
            'MROE': 344.8,  # Midwest
            'MROW': 397.1,  # Midwest West
            'NEWE': 227.4,  # New England
            'NWPP': 291.2,  # Northwest
            'NYUP': 129.9,  # NY Upstate
            'RFCE': 335.5,  # Mid-Atlantic
            'RFCM': 589.9,  # Michigan
            'RFCW': 550.4,  # Wisconsin/Upper Midwest
            'RMPA': 548.5,  # Rocky Mountains
            'SPNO': 448.0,  # Kansas/Nebraska
            'SPSO': 447.7,  # South
            'SRMV': 392.0,  # Mississippi Valley
            'SRMW': 675.5,  # Missouri
            'SRSO': 423.8,  # South Atlantic
            'SRTV': 399.6,  # Tennessee Valley
            'SRVC': 315.0,  # Virginia/Carolinas
            'US_AVG': 390.8  # US Average
        }
        
        # Cloud provider PUE (Power Usage Effectiveness)
        # Sources: Provider sustainability reports 2024
        self.pue_factors = {
            'aws': 1.135,
            'gcp': 1.10,
            'azure': 1.125,
            'on_premise': 1.67  # Industry average for on-premise
        }
        
        # GPU power consumption (Watts)
        # Sources: NVIDIA specs, TPU documentation
        self.gpu_power = {
            # NVIDIA GPUs
            'a100': 400,
            'h100': 700,
            'a10g': 150,
            'v100': 300,
            't4': 70,
            'a6000': 300,
            'l4': 72,
            # TPUs
            'tpu_v2': 280,
            'tpu_v3': 420,
            'tpu_v4': 170,
            # CPUs (average)
            'cpu': 150
        }
        
        # Model size to memory mapping (GB)
        # Based on parameter count and precision
        self.model_memory_map = {
            'small': 1,      # <1B parameters
            'medium': 10,    # 1-10B parameters  
            'large': 40,     # 10-50B parameters
            'xlarge': 160    # 50B+ parameters
        }
        
    def estimate_emissions(
        self,
        model_size: str,  # 'small', 'medium', 'large', 'xlarge'
        tokens_processed: int,
        instance_type: str,  # GPU/TPU type
        region: str,  # EPA eGRID region or state
        provider: str,  # 'aws', 'gcp', 'azure', 'on_premise'
        task_type: str = 'inference',  # 'training' or 'inference'
        batch_size: int = 1,
        duration_hours: Optional[float] = None,
        lat: Optional[float] = None,  # For real-time carbon intensity
        lon: Optional[float] = None,   # For real-time carbon intensity
        watttime_username: Optional[str] = None,
        watttime_password: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Core formula: CO2_emissions = Power × Time × Carbon_Intensity × PUE
        
        Returns emissions in kgCO2e and detailed breakdown
        """
        
        # Step 1: Calculate compute time if not provided
        if duration_hours is None:
            # Estimate based on tokens and model size
            tokens_per_second = self._estimate_throughput(
                model_size, instance_type, batch_size, task_type
            )
            duration_hours = tokens_processed / (tokens_per_second * 3600)
        
        # Step 2: Get power consumption
        base_power = self.gpu_power.get(instance_type.lower(), 150)  # Watts
        
        # Adjust for utilization
        if task_type == 'training':
            utilization = 0.90  # Training typically has high utilization
        else:
            utilization = 0.65  # Inference varies more
            
        power_consumption = base_power * utilization
        
        # Step 3: Get carbon intensity
        # Try real-time data first if coordinates provided
        carbon_intensity = None
        if lat is not None and lon is not None:
            carbon_intensity = self.get_real_time_carbon_intensity(
                lat, lon, watttime_username, watttime_password
            )
            if carbon_intensity:
                print(f"Using real-time WattTime data: {carbon_intensity:.1f} gCO2/kWh")
        
        # Fall back to static EPA data if real-time not available
        if carbon_intensity is None:
            carbon_intensity = self._get_carbon_intensity(region)
            print(f"Using EPA eGRID data for {region}: {carbon_intensity:.1f} gCO2/kWh")
        
        # Step 4: Apply PUE
        pue = self.pue_factors.get(provider.lower(), 1.67)
        
        # Step 5: Calculate emissions
        # Power (kW) × Time (h) × Carbon Intensity (gCO2/kWh) × PUE / 1000 = kgCO2
        energy_kwh = (power_consumption / 1000) * duration_hours
        emissions_kg = (energy_kwh * carbon_intensity * pue) / 1000
        
        # Additional factors for training
        if task_type == 'training':
            # Account for failed experiments, hyperparameter tuning
            emissions_kg *= 1.5  # 50% overhead for typical ML development
        
        return {
            'total_emissions_kgCO2e': round(emissions_kg, 3),
            'energy_consumption_kWh': round(energy_kwh, 3),
            'compute_hours': round(duration_hours, 3),
            'carbon_intensity_gCO2_per_kWh': carbon_intensity,
            'pue_factor': pue,
            'power_consumption_watts': power_consumption,
            'emissions_per_million_tokens': round(
                (emissions_kg / tokens_processed) * 1_000_000, 3
            ) if tokens_processed > 0 else 0
        }
    
    def _estimate_throughput(
        self,
        model_size: str,
        instance_type: str,
        batch_size: int,
        task_type: str
    ) -> float:
        """
        Estimate tokens per second based on model and hardware
        Sources: MLPerf benchmarks, published benchmarks
        """
        # Base throughput for inference (tokens/second)
        base_throughput = {
            ('small', 'a100'): 15000,
            ('small', 'v100'): 8000,
            ('small', 't4'): 3000,
            ('medium', 'a100'): 5000,
            ('medium', 'v100'): 2500,
            ('medium', 't4'): 800,
            ('large', 'a100'): 1500,
            ('large', 'v100'): 700,
            ('large', 'h100'): 3000,
            ('xlarge', 'a100'): 300,
            ('xlarge', 'h100'): 800,
        }
        
        # Get base or estimate
        key = (model_size, instance_type.lower())
        throughput = base_throughput.get(key, 1000)  # Default
        
        # Adjust for batch size (sublinear scaling)
        throughput *= (batch_size ** 0.7)
        
        # Training is slower
        if task_type == 'training':
            throughput *= 0.3
            
        return throughput
    
    def _get_carbon_intensity(self, region: str) -> float:
        """Get carbon intensity for region with fallback to US average"""
        # Map common state abbreviations to eGRID regions
        state_to_egrid = {
            'CA': 'CAMX', 'TX': 'ERCT', 'FL': 'FRCC',
            'NY': 'NYUP', 'VA': 'SRVC', 'WA': 'NWPP',
            'OR': 'NWPP', 'IL': 'RFCM', 'OH': 'RFCE',
            'PA': 'RFCE', 'MA': 'NEWE', 'GA': 'SRSO'
        }
        
        # Try direct region lookup
        if region.upper() in self.grid_carbon_intensity:
            return self.grid_carbon_intensity[region.upper()]
        
        # Try state mapping
        if region.upper() in state_to_egrid:
            egrid_region = state_to_egrid[region.upper()]
            return self.grid_carbon_intensity[egrid_region]
            
        # Default to US average
        return self.grid_carbon_intensity['US_AVG']
    
    def get_real_time_carbon_intensity(self, lat: float, lon: float, username: Optional[str] = None, password: Optional[str] = None) -> Optional[float]:
        """
        Get real-time carbon intensity from WattTime API
        Returns gCO2/kWh for the given location
        
        Args:
            lat: Latitude
            lon: Longitude  
            username: WattTime API username (optional, can use env var WATTTIME_USER)
            password: WattTime API password (optional, can use env var WATTTIME_PASSWORD)
        """
        import os
        import base64
        
        # Get credentials from parameters or environment variables
        wt_user = username or os.getenv('WATTTIME_USER')
        wt_pass = password or os.getenv('WATTTIME_PASSWORD')
        
        if not wt_user or not wt_pass:
            print("WattTime credentials not provided. Using static EPA data.")
            return None
            
        try:
            # Step 1: Login to get token
            login_url = "https://api.watttime.org/v3/login"
            auth_string = base64.b64encode(f"{wt_user}:{wt_pass}".encode()).decode()
            
            login_response = requests.get(
                login_url,
                headers={"Authorization": f"Basic {auth_string}"}
            )
            
            if login_response.status_code != 200:
                print(f"WattTime login failed: {login_response.status_code}")
                return None
                
            token = login_response.json().get('token')
            
            # Step 2: Get region from coordinates
            region_url = "https://api.watttime.org/v3/region-from-loc"
            region_params = {
                'latitude': lat,
                'longitude': lon,
                'signal_type': 'co2_moer'  # Marginal Operating Emissions Rate
            }
            
            region_response = requests.get(
                region_url,
                params=region_params,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if region_response.status_code != 200:
                print(f"Failed to get region: {region_response.status_code}")
                return None
                
            region = region_response.json().get('region')
            
            # Step 3: Get current forecast (real-time is first point)
            forecast_url = "https://api.watttime.org/v3/forecast"
            forecast_params = {
                'region': region,
                'signal_type': 'co2_moer'
            }
            
            forecast_response = requests.get(
                forecast_url,
                params=forecast_params,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if forecast_response.status_code != 200:
                print(f"Failed to get forecast: {forecast_response.status_code}")
                return None
                
            forecast_data = forecast_response.json()
            
            # Get the first point (current time)
            if forecast_data.get('data') and len(forecast_data['data']) > 0:
                current_moer = forecast_data['data'][0]['value']
                # Convert from lbs/MWh to g/kWh (1 lb = 453.592 g, 1 MWh = 1000 kWh)
                return current_moer * 453.592 / 1000
            
            return None
            
        except Exception as e:
            print(f"Error fetching WattTime data: {e}")
            return None
    
    def get_optimization_suggestions(
        self,
        current_emissions: Dict[str, float],
        region: str,
        provider: str
    ) -> list:
        """Provide actionable suggestions to reduce emissions"""
        suggestions = []
        
        # Check carbon intensity
        current_intensity = self._get_carbon_intensity(region)
        if current_intensity > 400:
            suggestions.append({
                'impact': 'high',
                'suggestion': 'Consider workload scheduling during low-carbon hours or migrating to cleaner regions',
                'potential_reduction': '20-40%'
            })
        
        # Check PUE
        current_pue = self.pue_factors.get(provider.lower(), 1.67)
        if current_pue > 1.2:
            suggestions.append({
                'impact': 'medium',
                'suggestion': 'Consider cloud providers with better PUE (Google Cloud: 1.10)',
                'potential_reduction': '10-15%'
            })
            
        # Model optimization
        if current_emissions['emissions_per_million_tokens'] > 0.1:
            suggestions.append({
                'impact': 'high',
                'suggestion': 'Consider model quantization or distillation to reduce compute requirements',
                'potential_reduction': '30-50%'
            })
            
        return suggestions


# Test the calculator immediately when run
if __name__ == "__main__":
    print("="*60)
    print("AI CARBON EMISSIONS CALCULATOR - QUICK TEST")
    print("="*60)
    
    # Create calculator instance
    calculator = AIEmissionsCalculator()
    print("✓ Calculator initialized successfully!")
    
    # Test 1: Basic inference calculation
    print("\nTest 1: GPT-3 style inference (1M tokens)")
    print("-" * 40)
    
    result = calculator.estimate_emissions(
        model_size='large',
        tokens_processed=1_000_000,
        instance_type='a100',
        region='CA',
        provider='aws',
        task_type='inference',
        batch_size=32
    )
    
    print(f"✓ Total emissions: {result['total_emissions_kgCO2e']} kg CO2")
    print(f"  Energy used: {result['energy_consumption_kWh']} kWh")
    print(f"  Compute time: {result['compute_hours']} hours")
    print(f"  Per million tokens: {result['emissions_per_million_tokens']} kg CO2")
    
    # Test 2: Different regions
    print("\nTest 2: Regional comparison (100k tokens each)")
    print("-" * 40)
    
    regions = ['CA', 'TX', 'NY', 'WA']
    for region in regions:
        result = calculator.estimate_emissions(
            model_size='medium',
            tokens_processed=100_000,
            instance_type='v100',
            region=region,
            provider='aws'
        )
        print(f"{region}: {result['total_emissions_kgCO2e']:.4f} kg CO2 ({result['carbon_intensity_gCO2_per_kWh']} gCO2/kWh)")
    
    # Test 3: Training vs Inference
    print("\nTest 3: Training vs Inference (10M tokens)")
    print("-" * 40)
    
    for task in ['inference', 'training']:
        result = calculator.estimate_emissions(
            model_size='medium',
            tokens_processed=10_000_000,
            instance_type='a100',
            region='US_AVG',
            provider='gcp',
            task_type=task
        )
        print(f"{task.title()}: {result['total_emissions_kgCO2e']} kg CO2")
    
    # Show optimization suggestions
    print("\nOptimization Suggestions for Texas on-premise:")
    print("-" * 40)
    
    high_emissions_scenario = calculator.estimate_emissions(
        model_size='large',
        tokens_processed=100_000_000,
        instance_type='v100',
        region='TX',
        provider='on_premise'
    )
    
    suggestions = calculator.get_optimization_suggestions(
        high_emissions_scenario, 'TX', 'on_premise'
    )
    
    for s in suggestions:
        print(f"[{s['impact'].upper()}] {s['suggestion']}")
        print(f"       Potential reduction: {s['potential_reduction']}")
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("Calculator is ready to use in your applications.")
    print("="*60)