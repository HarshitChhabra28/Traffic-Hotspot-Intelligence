import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pydeck as pdk

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Miami Traffic Intelligence", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Miami_Accidents_Cleaned.csv')
        
        # Clean Coordinates
        df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
        df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
        df = df.dropna(subset=['Start_Lat', 'Start_Lng'])
        
        # Clean Time
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['Hour'] = df['Start_Time'].dt.hour
        
        # Feature Engineering: Severity
        df['Is_Major'] = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)
        
        return df
    except Exception as e:
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.error("❌ Error: Dataset not found or empty.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Traffic Filters")

# Time Filter
time_filter = st.sidebar.radio(
    "Select Time Period:",
    ["All Day", "Morning Rush (6-9 AM)", "Evening Rush (4-7 PM)", "Night Owl (10 PM - 4 AM)"]
)

# Algorithm Parameters
st.sidebar.subheader("🤖 Precision Tuning")
# Default: 65m (Scientific Optimal)
distance_meters = st.sidebar.slider("Radius (Meters)", 20, 500, 65, 5) 
min_samples = st.sidebar.slider("Min Accidents per Cluster", 5, 100, 15)

# --- FILTER LOGIC ---
if time_filter == "Morning Rush (6-9 AM)":
    df = df_raw[(df_raw['Hour'] >= 6) & (df_raw['Hour'] <= 9)]
elif time_filter == "Evening Rush (4-7 PM)":
    df = df_raw[(df_raw['Hour'] >= 16) & (df_raw['Hour'] <= 19)]
elif time_filter == "Night Owl (10 PM - 4 AM)":
    df = df_raw[(df_raw['Hour'] >= 22) | (df_raw['Hour'] <= 4)]
else:
    df = df_raw

# --- MAIN DASHBOARD ---
st.title("🚦 Miami Accident Hotspots")

if len(df) > 0:
    # --- 2. RUN DBSCAN ---
    kms_per_radian = 6371.0088
    epsilon = (distance_meters / 1000) / kms_per_radian
    coords = df[['Start_Lat', 'Start_Lng']].to_numpy()
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df['cluster'] = db.labels_
    
    # Filter Noise
    hotspots_df = df[df['cluster'] != -1].copy()
    n_clusters = len(set(hotspots_df['cluster']))
    
    if n_clusters > 0:
        # --- 3. CALCULATE CENTROIDS (The "Bullet Hole" Logic) ---
        # Group by cluster to find the exact center of each hotspot
        centroids = hotspots_df.groupby('cluster').agg(
            latitude=('Start_Lat', 'mean'),       
            longitude=('Start_Lng', 'mean'),      
            Severe_Accidents=('Is_Major', 'sum'), 
            Total_Accidents=('cluster', 'count'), 
            Street=('Street', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        ).reset_index()
        
        # --- 4. VISUALIZATION (PyDeck Map) ---
        col1, col2 = st.columns(2)
        col1.metric("🔥 Hotspots Identified", len(centroids))
        col2.metric("🚗 Accidents Analyzed", len(hotspots_df))

        # Define the Layer (Tight Red Circles)
        layer = pdk.Layer(
            "ScatterplotLayer",
            centroids,
            get_position=["longitude", "latitude"],
            get_color=[255, 0, 0, 200], # Red
            get_radius=30,              # FIXED SIZE: 30m radius for sharp bullet holes
            pickable=True,
        )

        # Define the View
        view_state = pdk.ViewState(
            latitude=centroids["latitude"].mean(),
            longitude=centroids["longitude"].mean(),
            zoom=11,
            pitch=0,
        )

        # Tooltip (Hover info)
        tooltip = {
            "html": "<b>{Street}</b><br/>Total Crashes: {Total_Accidents}<br/>Severe: {Severe_Accidents}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

        # --- 5. LEADERBOARD ---
        st.markdown("### 🏆 Top 5 Most Dangerous Intersections")
        leaderboard = centroids.sort_values('Severe_Accidents', ascending=False).head(5)
        st.table(leaderboard[['Street', 'Total_Accidents', 'Severe_Accidents']])

        # Dynamic Insight
        top_street = leaderboard.iloc[0]['Street']
        st.info(f"💡 **Insight:** The #1 high-risk location is **{top_street}**. Resources should be prioritized here.")

    else:
        st.warning("⚠️ No clusters found! Try increasing Radius.")
        # Show raw noise for context
        st.markdown("**Showing raw accident data (Gray) instead:**")
        st.map(df[['Start_Lat', 'Start_Lng']].rename(columns={'Start_Lat': 'latitude', 'Start_Lng': 'longitude'}), color='#cccccc')

else:
    st.error("No data available for this time range.")