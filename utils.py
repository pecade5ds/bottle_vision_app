import geocoder
import requests

def filter_and_count(data, threshold=0.5, class_var="class"):
    filtered_data = [item for item in data if item["confidence"] >= threshold]
    result = {}
    for item in filtered_data:
        class_name = item[class_var]
        result[class_name] = result.get(class_name, 0) + 1
    return result


def get_location_geocoder() -> Tuple[Optional[float], Optional[float]]:
    """
    Get location using geocoder library
    """
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng[0], g.latlng[1]
    return None, None

def get_location_ipapi(st) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback method using ipapi.co service
    """
    try:
        response = requests.get('https://ipapi.co/json/')
        if response.status_code == 200:
            data = response.json()
            lat = data.get('latitude')
            lon = data.get('longitude')
            
            if lat is not None and lon is not None:
                # Store additional location data in session state
                st.session_state.location_data = {
                    'city': data.get('city'),
                    'region': data.get('region'),
                    'country': data.get('country_name'),
                    'ip': data.get('ip')
                }
                return lat, lon
    except requests.RequestException as e:
        st.error(f"Error retrieving location from ipapi.co: {str(e)}")
    return None, None

def get_location(st) -> Tuple[Optional[float], Optional[float]]:
    """
    Tries to get location first using geocoder, then falls back to ipapi.co
    """
    # Try geocoder first
    lat, lon = get_location_geocoder()
    
    # If geocoder fails, try ipapi
    if lat is None:
        st.info("Primary geolocation method unsuccessful, trying alternative...")
        lat, lon = get_location_ipapi(st)
    
    return lat, lon
