"""
Geocoder Module for Bangalore Real Estate Prediction
=====================================================
Maps Bangalore location names to latitude/longitude coordinates.
Uses mock coordinates based on real Bangalore geography.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


# Bangalore center coordinates
BANGALORE_CENTER = (12.9716, 77.5946)

# Major area coordinates (real approximate locations)
BANGALORE_LOCATIONS = {
    # Central Bangalore
    'Mg Road': (12.9756, 77.6069),
    'Brigade Road': (12.9714, 77.6062),
    'Koramangala': (12.9349, 77.6175),
    'Indiranagar': (12.9719, 77.6412),
    'Jayanagar': (12.9250, 77.5820),
    'Basavanagudi': (12.9430, 77.5740),
    'Malleshwaram': (12.9969, 77.5687),
    'Rajajinagar': (12.9890, 77.5520),
    
    # East Bangalore
    'Whitefield': (12.9698, 77.7500),
    'Marathahalli': (12.9591, 77.6971),
    'Kr Puram': (12.9983, 77.6882),
    'Mahadevapura': (12.9915, 77.6808),
    'Brookefield': (12.9706, 77.7166),
    'Kadugodi': (12.9891, 77.7641),
    'Hoodi': (12.9913, 77.7138),
    'Varthur': (12.9386, 77.7502),
    'Bellandur': (12.9261, 77.6798),
    'Hsr Layout': (12.9122, 77.6414),
    'Btm Layout': (12.9165, 77.6115),
    
    # South Bangalore
    'Electronic City': (12.8399, 77.6770),
    'Electronic City Phase Ii': (12.8457, 77.6593),
    'Electronics City Phase 1': (12.8453, 77.6595),
    'Bannerghatta Road': (12.8879, 77.5967),
    'Jp Nagar': (12.9064, 77.5857),
    '7Th Phase Jp Nagar': (12.8954, 77.5915),
    '8Th Phase Jp Nagar': (12.8870, 77.5860),
    'Uttarahalli': (12.9066, 77.5440),
    'Kanakapura Road': (12.8755, 77.5510),
    'Sarjapur Road': (12.9087, 77.6857),
    'Sarjapur': (12.8580, 77.7870),
    'Begur Road': (12.8700, 77.6260),
    'Begur': (12.8686, 77.6206),
    'Bommanahalli': (12.8945, 77.6150),
    'Arekere': (12.8857, 77.6030),
    'Hulimavu': (12.8781, 77.5988),
    'Gottigere': (12.8609, 77.5859),
    'Banashankari': (12.9266, 77.5581),
    'Banashankari Stage Iii': (12.9150, 77.5420),
    
    # North Bangalore
    'Hebbal': (13.0358, 77.5970),
    'Yelahanka': (13.1005, 77.5940),
    'Thanisandra': (13.0550, 77.6320),
    'Hennur Road': (13.0300, 77.6420),
    'Hennur': (13.0350, 77.6380),
    'Devanahalli': (13.2467, 77.7120),
    'Jakkur': (13.0706, 77.6100),
    'Sahakara Nagar': (13.0620, 77.5780),
    'Rmv Extension': (13.0164, 77.5779),
    'Rt Nagar': (13.0218, 77.5960),
    'Kalyan Nagar': (13.0220, 77.6380),
    'Hrbr Layout': (13.0154, 77.6500),
    'Banaswadi': (13.0100, 77.6480),
    
    # West Bangalore
    'Yeshwanthpur': (13.0220, 77.5430),
    'Rajaji Nagar': (12.9890, 77.5520),
    'Vijayanagar': (12.9700, 77.5350),
    'Nagarbhavi': (12.9610, 77.5110),
    'Kengeri': (12.9100, 77.4820),
    'Rajarajeshwari Nagar': (12.9150, 77.5070),
    'Mysore Road': (12.9320, 77.5130),
    'Magadi Road': (12.9680, 77.5120),
    
    # Additional popular areas
    'Indira Nagar': (12.9719, 77.6412),
    'Domlur': (12.9610, 77.6380),
    'Old Airport Road': (12.9610, 77.6480),
    'Cv Raman Nagar': (12.9850, 77.6630),
    'Kasturi Nagar': (12.9926, 77.6576),
    'Hormavu': (13.0330, 77.6380),
    'Nagavara': (13.0450, 77.6180),
    'Ramamurthy Nagar': (13.0120, 77.6680),
    'Tin Factory': (13.0050, 77.6550),
    'Ulsoor': (12.9830, 77.6180),
    'Langford Town': (12.9490, 77.5990),
    'Richmond Town': (12.9620, 77.6000),
    'Shantinagar': (12.9500, 77.5980),
    'Wilson Garden': (12.9380, 77.5930),
    'Chamrajpet': (12.9580, 77.5670),
    'Chickpete': (12.9670, 77.5740),
    'Majestic': (12.9770, 77.5710),
}


def generate_mock_coordinates(location: str, seed: int = None) -> tuple:
    """
    Generate mock coordinates for a location within Bangalore bounds.
    
    Uses the location name as seed for consistent coordinates.
    """
    if seed is None:
        seed = hash(location.lower()) % (2**32)
    
    np.random.seed(seed)
    
    # Bangalore bounds (approximate)
    lat_min, lat_max = 12.75, 13.25
    lng_min, lng_max = 77.35, 77.85
    
    # Generate coordinates with some clustering around center
    lat = BANGALORE_CENTER[0] + np.random.normal(0, 0.08)
    lng = BANGALORE_CENTER[1] + np.random.normal(0, 0.08)
    
    # Clamp to bounds
    lat = max(lat_min, min(lat_max, lat))
    lng = max(lng_min, min(lng_max, lng))
    
    return (round(lat, 6), round(lng, 6))


def geocode_location(location: str) -> tuple:
    """
    Get coordinates for a Bangalore location.
    
    First checks the known locations dictionary, then generates
    consistent mock coordinates for unknown locations.
    """
    if pd.isna(location):
        return BANGALORE_CENTER
    
    location_clean = str(location).strip().title()
    
    # Check known locations
    if location_clean in BANGALORE_LOCATIONS:
        return BANGALORE_LOCATIONS[location_clean]
    
    # Check partial matches
    for known_loc, coords in BANGALORE_LOCATIONS.items():
        if known_loc in location_clean or location_clean in known_loc:
            return coords
    
    # Generate consistent mock coordinates
    return generate_mock_coordinates(location_clean)


def add_coordinates(df: pd.DataFrame, location_column: str = 'location_clean') -> pd.DataFrame:
    """
    Add latitude and longitude columns to DataFrame.
    """
    df = df.copy()
    
    print("Geocoding locations...")
    coords = df[location_column].apply(geocode_location)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    print(f"  ✓ Geocoded {len(df):,} properties")
    print(f"  Lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    print(f"  Lng range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    
    return df


def get_nearby_properties(df: pd.DataFrame, lat: float, lng: float, 
                          radius_km: float = 2.0, limit: int = 10) -> pd.DataFrame:
    """
    Get properties within a radius of given coordinates.
    
    Uses Haversine formula for distance calculation.
    """
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
    
    df = df.copy()
    df['distance_km'] = df.apply(
        lambda row: haversine(lat, lng, row['latitude'], row['longitude']), 
        axis=1
    )
    
    nearby = df[df['distance_km'] <= radius_km].nsmallest(limit, 'distance_km')
    
    return nearby


if __name__ == "__main__":
    # Test geocoding
    test_locations = [
        'Whitefield',
        'Electronic City',
        'Koramangala',
        'Unknown Area XYZ'
    ]
    
    for loc in test_locations:
        coords = geocode_location(loc)
        print(f"{loc}: {coords}")
