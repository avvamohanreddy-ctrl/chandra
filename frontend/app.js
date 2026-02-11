/**
 * Bangalore Real Estate Price Predictor - Frontend Application
 * ============================================================
 * Handles map initialization, form submission, API calls,
 * and result visualization.
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    BANGALORE_CENTER: [12.9716, 77.5946],
    DEFAULT_ZOOM: 11,
    TILE_URL: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    TILE_ATTRIBUTION: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
};

// State
let map = null;
let markers = [];
let landmarkMarkers = [];
let predictionMarker = null;
let routingControl = null;
let currentPropertyCoords = null;

// ============================================================================
// MAP INITIALIZATION
// ============================================================================

function initMap() {
    // Create map centered on Bangalore
    map = L.map('map', {
        center: CONFIG.BANGALORE_CENTER,
        zoom: CONFIG.DEFAULT_ZOOM,
        zoomControl: true
    });

    // Add dark tile layer
    L.tileLayer(CONFIG.TILE_URL, {
        attribution: CONFIG.TILE_ATTRIBUTION,
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Add initial marker at center
    addCenterMarker();

    console.log('🗺️ Map initialized');
}

function addCenterMarker() {
    const pulsingIcon = L.divIcon({
        className: 'pulsing-marker',
        html: `
            <div style="
                width: 20px;
                height: 20px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                border-radius: 50%;
                border: 3px solid white;
                box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
            "></div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    predictionMarker = L.marker(CONFIG.BANGALORE_CENTER, { icon: pulsingIcon })
        .addTo(map)
        .bindPopup(createPopupContent({
            title: 'Bangalore City Center',
            details: 'Enter property details to get a price prediction'
        }));
}

// ============================================================================
// LOCATIONS AUTOCOMPLETE
// ============================================================================

async function loadLocations() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/locations`);
        if (!response.ok) throw new Error('Failed to fetch locations');

        const locations = await response.json();
        const datalist = document.getElementById('locations-list');

        locations.forEach(loc => {
            const option = document.createElement('option');
            option.value = loc.name;
            datalist.appendChild(option);
        });

        console.log(`📍 Loaded ${locations.length} locations`);
    } catch (error) {
        console.warn('Could not load locations:', error.message);
        // Add some default locations
        const defaultLocations = [
            'Whitefield', 'Koramangala', 'Electronic City', 'HSR Layout',
            'Marathahalli', 'Indiranagar', 'Jayanagar', 'Hebbal',
            'Sarjapur Road', 'Bellandur', 'JP Nagar', 'Bannerghatta Road'
        ];

        const datalist = document.getElementById('locations-list');
        defaultLocations.forEach(loc => {
            const option = document.createElement('option');
            option.value = loc;
            datalist.appendChild(option);
        });
    }
}

// ============================================================================
// PREDICTION
// ============================================================================

async function predictPrice(formData) {
    showLoading(true);

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        updateMap(result);

        console.log('✅ Prediction successful:', result);
    } catch (error) {
        console.error('Prediction error:', error);

        // Demo mode: generate mock prediction
        const mockResult = generateMockPrediction(formData);
        displayResults(mockResult);
        updateMap(mockResult);
    } finally {
        showLoading(false);
    }
}

function generateMockPrediction(formData) {
    // Mock coordinates for demo
    const mockCoords = getMockCoordinates(formData.location);

    // Mock price calculation
    const basePrice = 6500;
    const bhkMultiplier = 1 + (formData.bhk - 2) * 0.15;
    const sizeMultiplier = formData.total_sqft < 1000 ? 1.1 : formData.total_sqft > 2000 ? 0.9 : 1;

    const pricePerSqft = basePrice * bhkMultiplier * sizeMultiplier + (Math.random() - 0.5) * 1000;
    const totalPrice = pricePerSqft * formData.total_sqft;

    return {
        success: true,
        location: formData.location,
        coordinates: { latitude: mockCoords[0], longitude: mockCoords[1] },
        predicted_price_per_sqft: Math.round(pricePerSqft * 100) / 100,
        total_estimated_price: Math.round(totalPrice * 100) / 100,
        total_estimated_price_formatted: `₹${(totalPrice / 100000).toFixed(2)} Lakhs`,
        confidence_interval: {
            lower: Math.round(pricePerSqft * 0.85 * 100) / 100,
            upper: Math.round(pricePerSqft * 1.15 * 100) / 100
        },
        nearby_comparables: generateMockComparables(mockCoords, formData.bhk),
        nearby_landmarks: generateMockLandmarks(mockCoords)
    };
}

function generateMockLandmarks(coords) {
    const landmarks = [
        { name: 'MG Road Metro', type: 'metro', icon: '🚇', lat: 12.9756, lng: 77.6063 },
        { name: 'Majestic Metro', type: 'metro', icon: '🚇', lat: 12.9772, lng: 77.5662 },
        { name: 'Bangalore City Junction', type: 'railway', icon: '🚂', lat: 12.9791, lng: 77.5713 },
        { name: 'Phoenix Marketcity', type: 'mall', icon: '🛒', lat: 12.9969, lng: 77.6970 },
        { name: 'Forum Mall', type: 'mall', icon: '🛒', lat: 12.9344, lng: 77.6104 },
        { name: 'Manipal Hospital', type: 'hospital', icon: '🏥', lat: 12.9588, lng: 77.6479 },
        { name: 'Manyata Tech Park', type: 'it_park', icon: '🏢', lat: 13.0474, lng: 77.6229 },
        { name: 'Electronic City Phase 1', type: 'it_park', icon: '🏢', lat: 12.8456, lng: 77.6605 },
        { name: 'Majestic Bus Station', type: 'bus', icon: '🚌', lat: 12.9771, lng: 77.5711 },
    ];

    // Calculate distance and direction for each landmark
    const result = landmarks.map(lm => {
        const dist = haversineDistance(coords[0], coords[1], lm.lat, lm.lng);
        const dir = getBearing(coords[0], coords[1], lm.lat, lm.lng);
        return {
            name: lm.name,
            type: lm.type,
            icon: lm.icon,
            latitude: lm.lat,
            longitude: lm.lng,
            distance_km: Math.round(dist * 100) / 100,
            direction: dir,
            color: getLandmarkColor(lm.type)
        };
    }).filter(lm => lm.distance_km <= 8)  // Within 8km
        .sort((a, b) => a.distance_km - b.distance_km)
        .slice(0, 8);  // Top 8

    return result;
}

function haversineDistance(lat1, lng1, lat2, lng2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLng / 2) * Math.sin(dLng / 2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getBearing(lat1, lng1, lat2, lng2) {
    const dLng = (lng2 - lng1) * Math.PI / 180;
    lat1 = lat1 * Math.PI / 180;
    lat2 = lat2 * Math.PI / 180;
    const x = Math.cos(lat2) * Math.sin(dLng);
    const y = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
    let bearing = Math.atan2(x, y) * 180 / Math.PI;
    bearing = (bearing + 360) % 360;
    const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return dirs[Math.floor((bearing + 22.5) / 45) % 8];
}

function getLandmarkColor(type) {
    const colors = {
        metro: '#3b82f6',
        railway: '#ef4444',
        bus: '#f59e0b',
        mall: '#8b5cf6',
        hospital: '#10b981',
        it_park: '#6366f1'
    };
    return colors[type] || '#6b7280';
}

function getMockCoordinates(location) {
    const locationMap = {
        'Whitefield': [12.9698, 77.7500],
        'Koramangala': [12.9349, 77.6175],
        'Electronic City': [12.8399, 77.6770],
        'HSR Layout': [12.9122, 77.6414],
        'Marathahalli': [12.9591, 77.6971],
        'Indiranagar': [12.9719, 77.6412],
        'Jayanagar': [12.9250, 77.5820],
        'Hebbal': [13.0358, 77.5970],
        'Sarjapur Road': [12.9087, 77.6857],
        'Bellandur': [12.9261, 77.6798],
        'JP Nagar': [12.9064, 77.5857],
        'Bannerghatta Road': [12.8879, 77.5967]
    };

    // Find matching location
    const normalizedLoc = location.toLowerCase();
    for (const [key, coords] of Object.entries(locationMap)) {
        if (key.toLowerCase().includes(normalizedLoc) || normalizedLoc.includes(key.toLowerCase())) {
            return coords;
        }
    }

    // Random location near Bangalore center
    return [
        12.9716 + (Math.random() - 0.5) * 0.15,
        77.5946 + (Math.random() - 0.5) * 0.15
    ];
}

function generateMockComparables(coords, bhk) {
    const comparables = [];
    const locations = ['Near Location', 'Nearby Area', 'Close By', 'Adjacent'];

    for (let i = 0; i < 4; i++) {
        comparables.push({
            location: locations[i],
            bhk: bhk,
            total_sqft: 1000 + Math.round(Math.random() * 1000),
            price_per_sqft: 5000 + Math.round(Math.random() * 3000),
            distance_km: Math.round((0.5 + Math.random() * 2) * 10) / 10,
            latitude: coords[0] + (Math.random() - 0.5) * 0.02,
            longitude: coords[1] + (Math.random() - 0.5) * 0.02
        });
    }

    return comparables.sort((a, b) => a.distance_km - b.distance_km);
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayResults(result) {
    // Show results section
    document.getElementById('results').classList.remove('hidden');

    // Update values
    document.getElementById('total-price').textContent = result.total_estimated_price_formatted;
    document.getElementById('price-per-sqft').textContent = `₹${result.predicted_price_per_sqft.toLocaleString()} per sq.ft`;

    document.getElementById('ci-lower').textContent = `₹${result.confidence_interval.lower.toLocaleString()}`;
    document.getElementById('ci-upper').textContent = `₹${result.confidence_interval.upper.toLocaleString()}`;

    // Display comparables
    const comparablesContainer = document.getElementById('comparables-list');
    comparablesContainer.innerHTML = '';

    if (result.nearby_comparables && result.nearby_comparables.length > 0) {
        result.nearby_comparables.forEach((comp, index) => {
            const item = document.createElement('div');
            item.className = 'comparable-item';
            item.onclick = () => focusComparable(comp);

            item.innerHTML = `
                <div class="comparable-info">
                    <div class="comparable-location">${comp.location}</div>
                    <div class="comparable-details">${comp.bhk} BHK • ${comp.total_sqft.toLocaleString()} sq.ft • ${comp.distance_km} km</div>
                </div>
                <div class="comparable-price">₹${comp.price_per_sqft.toLocaleString()}</div>
            `;

            comparablesContainer.appendChild(item);
        });
    } else {
        comparablesContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No comparable properties found</p>';
    }

    // Display landmarks
    displayLandmarks(result.nearby_landmarks || []);
}

function displayLandmarks(landmarks) {
    const landmarksContainer = document.getElementById('landmarks-list');
    landmarksContainer.innerHTML = '';

    if (landmarks.length > 0) {
        landmarks.forEach(lm => {
            const item = document.createElement('div');
            item.className = 'landmark-item';

            // Check if transport (bus, railway, metro) - show directions button
            const isTransport = ['bus', 'railway', 'metro'].includes(lm.type);

            item.innerHTML = `
                <span class="landmark-icon">${lm.icon}</span>
                <div class="landmark-info">
                    <div class="landmark-name">${lm.name}</div>
                    <div class="landmark-distance">${lm.distance_km} km away</div>
                </div>
                ${isTransport
                    ? `<button class="directions-btn" title="Get Directions">🧭</button>`
                    : `<span class="landmark-direction">${lm.direction}</span>`
                }
            `;

            // Click handler
            item.onclick = (e) => {
                if (e.target.classList.contains('directions-btn')) {
                    e.stopPropagation();
                    showRouteToLandmark(lm);
                } else {
                    focusLandmark(lm);
                }
            };

            landmarksContainer.appendChild(item);
        });
    } else {
        landmarksContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No landmarks found nearby</p>';
    }
}

// ============================================================================
// MAP UPDATES
// ============================================================================

function updateMap(result) {
    // Clear existing markers
    clearMarkers();

    const coords = [result.coordinates.latitude, result.coordinates.longitude];

    // Store property coords for routing
    currentPropertyCoords = coords;

    // Add prediction marker
    const predictionIcon = L.divIcon({
        className: 'prediction-marker-icon',
        html: `
            <div style="
                width: 24px;
                height: 24px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                border-radius: 50%;
                border: 3px solid white;
                box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
                animation: pulse 2s infinite;
            "></div>
            <style>
                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
            </style>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
    });

    predictionMarker = L.marker(coords, { icon: predictionIcon })
        .addTo(map)
        .bindPopup(createPopupContent({
            title: result.location,
            price: result.total_estimated_price_formatted,
            pricePerSqft: `₹${result.predicted_price_per_sqft.toLocaleString()}/sq.ft`
        }))
        .openPopup();

    // Add comparable markers
    if (result.nearby_comparables) {
        result.nearby_comparables.forEach(comp => {
            const compIcon = L.divIcon({
                className: 'comparable-marker-icon',
                html: `
                    <div style="
                        width: 14px;
                        height: 14px;
                        background: #10b981;
                        border-radius: 50%;
                        border: 2px solid white;
                        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
                    "></div>
                `,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            const marker = L.marker([comp.latitude, comp.longitude], { icon: compIcon })
                .addTo(map)
                .bindPopup(createPopupContent({
                    title: comp.location,
                    details: `${comp.bhk} BHK • ${comp.total_sqft.toLocaleString()} sq.ft`,
                    pricePerSqft: `₹${comp.price_per_sqft.toLocaleString()}/sq.ft`,
                    distance: `${comp.distance_km} km away`
                }));

            markers.push(marker);
        });
    }

    // Add landmark markers
    if (result.nearby_landmarks) {
        result.nearby_landmarks.forEach(lm => {
            const lmIcon = L.divIcon({
                className: 'landmark-marker-icon',
                html: `
                    <div style="
                        font-size: 20px;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                    ">${lm.icon}</div>
                `,
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });

            const marker = L.marker([lm.latitude, lm.longitude], { icon: lmIcon })
                .addTo(map)
                .bindPopup(createPopupContent({
                    title: `${lm.icon} ${lm.name}`,
                    details: `${lm.type.replace('_', ' ').toUpperCase()}`,
                    distance: `${lm.distance_km} km ${lm.direction}`
                }));

            landmarkMarkers.push(marker);
        });
    }

    // Fit bounds to show all markers
    if (markers.length > 0) {
        const group = new L.featureGroup([predictionMarker, ...markers]);
        map.fitBounds(group.getBounds().pad(0.2));
    } else {
        map.setView(coords, 14);
    }
}

function createPopupContent(data) {
    let content = `<div class="popup-title">${data.title}</div>`;

    if (data.details) {
        content += `<div class="popup-details">${data.details}</div>`;
    }

    if (data.price) {
        content += `<div class="popup-price">${data.price}</div>`;
    }

    if (data.pricePerSqft) {
        content += `<div class="popup-details">${data.pricePerSqft}</div>`;
    }

    if (data.distance) {
        content += `<div class="popup-details" style="color: var(--text-muted);">${data.distance}</div>`;
    }

    return content;
}

function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];

    landmarkMarkers.forEach(marker => map.removeLayer(marker));
    landmarkMarkers = [];

    if (predictionMarker) {
        map.removeLayer(predictionMarker);
        predictionMarker = null;
    }

    clearRoute();
}

function clearRoute() {
    if (routingControl) {
        map.removeControl(routingControl);
        routingControl = null;
    }
}

function focusLandmark(lm) {
    // Check if this is a bus or railway station - show route
    const showRoute = ['bus', 'railway', 'metro'].includes(lm.type);

    if (showRoute && currentPropertyCoords) {
        showRouteToLandmark(lm);
    } else {
        map.setView([lm.latitude, lm.longitude], 15);

        landmarkMarkers.forEach(marker => {
            const pos = marker.getLatLng();
            if (Math.abs(pos.lat - lm.latitude) < 0.0001 && Math.abs(pos.lng - lm.longitude) < 0.0001) {
                marker.openPopup();
            }
        });
    }
}

function showRouteToLandmark(lm) {
    // Clear existing route
    clearRoute();

    if (!currentPropertyCoords) {
        console.warn('No property location set');
        return;
    }

    const start = L.latLng(currentPropertyCoords[0], currentPropertyCoords[1]);
    const end = L.latLng(lm.latitude, lm.longitude);

    // Create routing control
    routingControl = L.Routing.control({
        waypoints: [start, end],
        routeWhileDragging: false,
        addWaypoints: false,
        draggableWaypoints: false,
        fitSelectedRoutes: true,
        showAlternatives: false,
        lineOptions: {
            styles: [
                { color: '#6366f1', opacity: 0.8, weight: 6 },
                { color: '#8b5cf6', opacity: 0.5, weight: 10 }
            ]
        },
        createMarker: function () { return null; }, // Don't create default markers
        router: L.Routing.osrmv1({
            serviceUrl: 'https://router.project-osrm.org/route/v1'
        })
    }).addTo(map);

    // Style the directions panel
    routingControl.on('routesfound', function (e) {
        const routes = e.routes;
        if (routes.length > 0) {
            const route = routes[0];
            const distance = (route.summary.totalDistance / 1000).toFixed(1);
            const time = Math.round(route.summary.totalTime / 60);

            console.log(`Route to ${lm.name}: ${distance} km, ${time} min`);
        }
    });
}

function focusComparable(comp) {
    map.setView([comp.latitude, comp.longitude], 15);

    // Find and open the marker popup
    markers.forEach(marker => {
        const markerLatLng = marker.getLatLng();
        if (Math.abs(markerLatLng.lat - comp.latitude) < 0.0001 &&
            Math.abs(markerLatLng.lng - comp.longitude) < 0.0001) {
            marker.openPopup();
        }
    });
}

// ============================================================================
// UI HELPERS
// ============================================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

// ============================================================================
// FORM HANDLING
// ============================================================================

function handleFormSubmit(event) {
    event.preventDefault();

    const form = event.target;
    const formData = {
        location: form.location.value,
        total_sqft: parseFloat(form.total_sqft.value),
        bhk: parseInt(form.bhk.value),
        bath: parseInt(form.bath.value),
        balcony: parseInt(form.balcony.value),
        area_type: form.area_type.value
    };

    console.log('📊 Submitting prediction request:', formData);
    predictPrice(formData);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Bangalore Real Estate Predictor loaded');

    // Initialize map
    initMap();

    // Load locations
    loadLocations();

    // Set up form submission
    document.getElementById('prediction-form').addEventListener('submit', handleFormSubmit);
});
