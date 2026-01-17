import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import pydeck as pdk

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Miami Traffic Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 1rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Title Styling */
    h1 {
        color: #1e3c72;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #2a5298;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: #667eea;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Info Box */
    .stAlert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Table Styling */
    table {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
        padding: 1rem;
        text-align: left;
    }
    
    tbody tr:hover {
        background-color: #f5f7fa;
    }
    
    tbody tr td {
        padding: 0.8rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Radio Buttons */
    div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        transition: all 0.3s;
    }
    
    div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Warning/Error Boxes */
    .stWarning {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
    }
    
    .stError {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
    }
    
    /* Subheader in sidebar */
    [data-testid="stSidebar"] h3 {
        color: white !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

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
st.sidebar.markdown("## ⚙️ Traffic Filters")

# Time Filter with enhanced styling
time_filter = st.sidebar.radio(
    "🕐 Select Time Period:",
    ["All Day", "Morning Rush (6-9 AM)", "Evening Rush (4-7 PM)", "Night Owl (10 PM - 4 AM)"]
)

# Algorithm Parameters
st.sidebar.markdown("### 🤖 Precision Tuning")
st.sidebar.caption("Adjust clustering parameters for optimal hotspot detection")

distance_meters = st.sidebar.slider("📍 Search Radius (Meters)", 20, 500, 65, 5,
                                    help="Defines the geographic area to group accidents") 
min_samples = st.sidebar.slider("📊 Min Accidents per Cluster", 5, 100, 15,
                                help="Minimum number of accidents to form a hotspot")

# Add info section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.caption("This dashboard uses DBSCAN clustering to identify accident hotspots in Miami, helping prioritize traffic safety resources.")

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
st.markdown("<h1>🚦 Miami Traffic Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>Advanced Accident Hotspot Detection & Analysis</p>", unsafe_allow_html=True)

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
        # --- 3. CALCULATE CENTROIDS ---
        centroids = hotspots_df.groupby('cluster').agg(
            latitude=('Start_Lat', 'mean'),       
            longitude=('Start_Lng', 'mean'),      
            Severe_Accidents=('Is_Major', 'sum'), 
            Total_Accidents=('cluster', 'count'), 
            Street=('Street', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        ).reset_index()
        
        # --- 4. METRICS ROW ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔥 Hotspots", len(centroids))
        with col2:
            st.metric("🚗 Total Crashes", len(hotspots_df))
        with col3:
            st.metric("⚠️ Severe Incidents", int(centroids['Severe_Accidents'].sum()))
        with col4:
            severity_rate = (centroids['Severe_Accidents'].sum() / len(hotspots_df) * 100)
            st.metric("📈 Severity Rate", f"{severity_rate:.1f}%")
        
        st.markdown("---")
        
        # --- 5. VISUALIZATION ---
        st.markdown("### 🗺️ Interactive Hotspot Map")
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            centroids,
            get_position=["longitude", "latitude"],
            get_color=[255, 0, 0, 200],
            get_radius=30,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=centroids["latitude"].mean(),
            longitude=centroids["longitude"].mean(),
            zoom=11,
            pitch=0,
        )

        tooltip = {
            "html": "<b>📍 {Street}</b><br/>Total Crashes: {Total_Accidents}<br/>Severe: {Severe_Accidents}",
            "style": {"backgroundColor": "#1e3c72", "color": "white", "padding": "10px", "borderRadius": "8px"}
        }

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

        # --- 6. LEADERBOARD ---
        st.markdown("### 🏆 Top 5 Most Dangerous Locations")
        leaderboard = centroids.sort_values('Severe_Accidents', ascending=False).head(5)
        
        # Format for better display
        leaderboard_display = leaderboard[['Street', 'Total_Accidents', 'Severe_Accidents']].copy()
        leaderboard_display.columns = ['🛣️ Location', '📊 Total Crashes', '⚠️ Severe Crashes']
        leaderboard_display.index = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][:len(leaderboard_display)]
        
        st.table(leaderboard_display)

        # --- 7. INSIGHTS ---
        top_street = leaderboard.iloc[0]['Street']
        top_severe = int(leaderboard.iloc[0]['Severe_Accidents'])
        top_total = int(leaderboard.iloc[0]['Total_Accidents'])
        
        st.info(f"💡 **Key Insight:** The highest-risk location is **{top_street}** with {top_total} total accidents ({top_severe} severe). Immediate intervention recommended for this area.")

    else:
        st.warning("⚠️ No clusters detected with current parameters. Try increasing the search radius or decreasing minimum samples.")
        
        st.markdown("**Displaying raw accident data for reference:**")
        st.map(df[['Start_Lat', 'Start_Lng']].rename(columns={'Start_Lat': 'latitude', 'Start_Lng': 'longitude'}), color='#cccccc')

else:
    st.error("❌ No data available for the selected time range. Please choose a different time period.")