"""
Bangalore Landmarks Database
============================
Contains coordinates for key public landmarks in Bangalore:
- Metro Stations
- Railway Stations  
- Bus Stands
- Malls/Shopping Centers
- Hospitals
- IT Parks
"""

import numpy as np
from typing import List, Dict, Tuple

# Bangalore Landmarks categorized by type
LANDMARKS = {
    # Metro Stations (Namma Metro)
    "metro": [
        {"name": "Majestic Metro", "lat": 12.9772, "lng": 77.5662, "type": "metro"},
        {"name": "MG Road Metro", "lat": 12.9756, "lng": 77.6063, "type": "metro"},
        {"name": "Indiranagar Metro", "lat": 12.9784, "lng": 77.6408, "type": "metro"},
        {"name": "Whitefield Metro", "lat": 12.9698, "lng": 77.7500, "type": "metro"},
        {"name": "Yelachenahalli Metro", "lat": 12.8949, "lng": 77.5736, "type": "metro"},
        {"name": "Nagasandra Metro", "lat": 13.0451, "lng": 77.5095, "type": "metro"},
        {"name": "Baiyappanahalli Metro", "lat": 12.9913, "lng": 77.6559, "type": "metro"},
        {"name": "Vijayanagar Metro", "lat": 12.9679, "lng": 77.5369, "type": "metro"},
        {"name": "Jayanagar Metro", "lat": 12.9293, "lng": 77.5821, "type": "metro"},
        {"name": "Banashankari Metro", "lat": 12.9141, "lng": 77.5741, "type": "metro"},
        {"name": "RV Road Metro", "lat": 12.9347, "lng": 77.5784, "type": "metro"},
        {"name": "Kempegowda Metro (Majestic)", "lat": 12.9765, "lng": 77.5721, "type": "metro"},
        {"name": "Trinity Metro", "lat": 12.9720, "lng": 77.6197, "type": "metro"},
        {"name": "Halasuru Metro", "lat": 12.9817, "lng": 77.6225, "type": "metro"},
        {"name": "Cubbon Park Metro", "lat": 12.9765, "lng": 77.5912, "type": "metro"},
    ],
    
    # Railway Stations
    "railway": [
        {"name": "Bangalore City Junction (SBC)", "lat": 12.9791, "lng": 77.5713, "type": "railway"},
        {"name": "Yeshwanthpur Junction", "lat": 13.0282, "lng": 77.5391, "type": "railway"},
        {"name": "Bangalore Cantonment", "lat": 12.9958, "lng": 77.5996, "type": "railway"},
        {"name": "Krishnarajapuram", "lat": 13.0021, "lng": 77.6823, "type": "railway"},
        {"name": "Whitefield Railway Station", "lat": 12.9850, "lng": 77.7520, "type": "railway"},
        {"name": "Kengeri", "lat": 12.9091, "lng": 77.4862, "type": "railway"},
        {"name": "Yelahanka Junction", "lat": 13.1016, "lng": 77.5967, "type": "railway"},
        {"name": "Baiyappanahalli Terminal", "lat": 12.9880, "lng": 77.6560, "type": "railway"},
    ],
    
    # Bus Stands
    "bus": [
        {"name": "Majestic Bus Station (KSRTC)", "lat": 12.9771, "lng": 77.5711, "type": "bus"},
        {"name": "Shantinagar Bus Stand", "lat": 12.9536, "lng": 77.5953, "type": "bus"},
        {"name": "Banashankari Bus Stand", "lat": 12.9154, "lng": 77.5732, "type": "bus"},
        {"name": "Kempegowda Bus Station", "lat": 12.9765, "lng": 77.5641, "type": "bus"},
        {"name": "Shivajinagar Bus Stand", "lat": 12.9857, "lng": 77.6052, "type": "bus"},
        {"name": "Marathahalli Bus Stop", "lat": 12.9591, "lng": 77.6974, "type": "bus"},
        {"name": "Electronic City Bus Stop", "lat": 12.8458, "lng": 77.6603, "type": "bus"},
        {"name": "Whitefield Bus Stand", "lat": 12.9700, "lng": 77.7400, "type": "bus"},
        {"name": "Koramangala Bus Stand", "lat": 12.9341, "lng": 77.6225, "type": "bus"},
        {"name": "Hebbal Bus Stand", "lat": 13.0358, "lng": 77.5901, "type": "bus"},
    ],
    
    # Shopping Malls
    "mall": [
        {"name": "Phoenix Marketcity", "lat": 12.9969, "lng": 77.6970, "type": "mall"},
        {"name": "Orion Mall", "lat": 13.0105, "lng": 77.5562, "type": "mall"},
        {"name": "Forum Mall (Koramangala)", "lat": 12.9344, "lng": 77.6104, "type": "mall"},
        {"name": "Mantri Square Mall", "lat": 12.9915, "lng": 77.5707, "type": "mall"},
        {"name": "UB City Mall", "lat": 12.9716, "lng": 77.5972, "type": "mall"},
        {"name": "Garuda Mall", "lat": 12.9708, "lng": 77.6100, "type": "mall"},
        {"name": "VR Bengaluru", "lat": 12.9674, "lng": 77.7025, "type": "mall"},
        {"name": "Gopalan Arcade Mall", "lat": 12.9012, "lng": 77.5855, "type": "mall"},
        {"name": "Central Mall (JP Nagar)", "lat": 12.9102, "lng": 77.5850, "type": "mall"},
        {"name": "Elements Mall (Nagavara)", "lat": 13.0445, "lng": 77.6197, "type": "mall"},
        {"name": "Lulu Mall", "lat": 13.0352, "lng": 77.5015, "type": "mall"},
    ],
    
    # Hospitals
    "hospital": [
        {"name": "Manipal Hospital (HAL)", "lat": 12.9588, "lng": 77.6479, "type": "hospital"},
        {"name": "Apollo Hospital (Bannerghatta)", "lat": 12.8958, "lng": 77.5970, "type": "hospital"},
        {"name": "Fortis Hospital (Cunningham)", "lat": 13.0013, "lng": 77.5871, "type": "hospital"},
        {"name": "Columbia Asia (Hebbal)", "lat": 13.0402, "lng": 77.5865, "type": "hospital"},
        {"name": "St. John's Medical College", "lat": 12.9313, "lng": 77.6223, "type": "hospital"},
        {"name": "Narayana Health City", "lat": 12.8764, "lng": 77.6011, "type": "hospital"},
        {"name": "Vikram Hospital", "lat": 12.9356, "lng": 77.5713, "type": "hospital"},
        {"name": "NIMHANS", "lat": 12.9436, "lng": 77.5976, "type": "hospital"},
        {"name": "Sakra World Hospital", "lat": 12.9544, "lng": 77.7139, "type": "hospital"},
        {"name": "Aster CMI Hospital", "lat": 13.0491, "lng": 77.5932, "type": "hospital"},
    ],
    
    # IT Parks
    "it_park": [
        {"name": "Electronic City Phase 1", "lat": 12.8456, "lng": 77.6605, "type": "it_park"},
        {"name": "Manyata Tech Park", "lat": 13.0474, "lng": 77.6229, "type": "it_park"},
        {"name": "Bagmane Tech Park", "lat": 12.9697, "lng": 77.6958, "type": "it_park"},
        {"name": "RMZ Ecospace (Bellandur)", "lat": 12.9261, "lng": 77.6798, "type": "it_park"},
        {"name": "Embassy Tech Village", "lat": 12.9324, "lng": 77.6889, "type": "it_park"},
        {"name": "Prestige Tech Park", "lat": 12.9305, "lng": 77.6873, "type": "it_park"},
        {"name": "ITPL (Whitefield)", "lat": 12.9852, "lng": 77.7315, "type": "it_park"},
        {"name": "Global Tech Park", "lat": 12.9340, "lng": 77.6107, "type": "it_park"},
        {"name": "Embassy Golf Links", "lat": 12.9597, "lng": 77.6510, "type": "it_park"},
    ],
}

# Emoji icons for landmark types
LANDMARK_ICONS = {
    "metro": "🚇",
    "railway": "🚂",
    "bus": "🚌",
    "mall": "🛒",
    "hospital": "🏥",
    "it_park": "🏢",
}

# Color codes for map markers
LANDMARK_COLORS = {
    "metro": "#3b82f6",    # Blue
    "railway": "#ef4444",   # Red
    "bus": "#f59e0b",       # Orange
    "mall": "#8b5cf6",      # Purple
    "hospital": "#10b981",  # Green
    "it_park": "#6366f1",   # Indigo
}


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def get_bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> str:
    """Calculate compass direction from point 1 to point 2."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    
    dlng = lng2 - lng1
    x = np.cos(lat2) * np.sin(dlng)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
    
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    
    # Convert to compass direction
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) / 45) % 8
    
    return directions[idx]


def get_nearby_landmarks(lat: float, lng: float, 
                        radius_km: float = 5.0,
                        limit_per_type: int = 3) -> List[Dict]:
    """
    Get nearby landmarks from a given location.
    
    Args:
        lat: Latitude of the property
        lng: Longitude of the property
        radius_km: Search radius in kilometers
        limit_per_type: Max landmarks to return per category
        
    Returns:
        List of nearby landmarks with distance and direction
    """
    nearby = []
    
    for category, landmarks in LANDMARKS.items():
        category_landmarks = []
        
        for landmark in landmarks:
            distance = haversine_distance(lat, lng, landmark["lat"], landmark["lng"])
            
            if distance <= radius_km:
                direction = get_bearing(lat, lng, landmark["lat"], landmark["lng"])
                category_landmarks.append({
                    "name": landmark["name"],
                    "type": landmark["type"],
                    "latitude": landmark["lat"],
                    "longitude": landmark["lng"],
                    "distance_km": round(distance, 2),
                    "direction": direction,
                    "icon": LANDMARK_ICONS.get(landmark["type"], "📍"),
                    "color": LANDMARK_COLORS.get(landmark["type"], "#6b7280"),
                })
        
        # Sort by distance and limit
        category_landmarks.sort(key=lambda x: x["distance_km"])
        nearby.extend(category_landmarks[:limit_per_type])
    
    # Sort all by distance
    nearby.sort(key=lambda x: x["distance_km"])
    
    return nearby


def get_all_landmarks() -> List[Dict]:
    """Get all landmarks for map display."""
    all_landmarks = []
    
    for category, landmarks in LANDMARKS.items():
        for landmark in landmarks:
            all_landmarks.append({
                "name": landmark["name"],
                "type": landmark["type"],
                "latitude": landmark["lat"],
                "longitude": landmark["lng"],
                "icon": LANDMARK_ICONS.get(landmark["type"], "📍"),
                "color": LANDMARK_COLORS.get(landmark["type"], "#6b7280"),
            })
    
    return all_landmarks


if __name__ == "__main__":
    # Test
    print("Testing landmarks module...")
    
    # Test from Koramangala
    lat, lng = 12.9349, 77.6175
    nearby = get_nearby_landmarks(lat, lng, radius_km=5)
    
    print(f"\nNearby landmarks from Koramangala ({lat}, {lng}):")
    for lm in nearby[:10]:
        print(f"  {lm['icon']} {lm['name']}: {lm['distance_km']} km {lm['direction']}")
    
    print(f"\nTotal landmarks in database: {sum(len(v) for v in LANDMARKS.values())}")
