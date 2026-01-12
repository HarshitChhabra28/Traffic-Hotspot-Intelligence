🚦 Miami Urban Traffic Intelligence System

🔍 Project Overview

Goal: Optimize city resource allocation by identifying high-risk traffic zones using Unsupervised Machine Learning.

City planners often struggle to differentiate between random "human error" accidents and structural road failures. This project utilizes DBSCAN Clustering on geospatial data (derived from the US Accidents dataset) to filter out noise and pinpoint specific intersections ("Bullet Holes") that require safety interventions.

💡 Key Intelligence & Results

1. The "Severity Paradox" (Crucial Finding)

My analysis revealed a critical divergence between Frequency (How many accidents) and Severity (How bad they are).

The Data: While the Palmetto Expressway has the highest volume of accidents (12,510), I-95 South segments proved significantly more hazardous with a 37% major accident rate (Severity 3+) compared to Palmetto's 6%.

The Insight: Volume does not equal danger. Resources should be prioritized based on severity, not just count.

2. The "Morning vs. Afternoon" Split

Using temporal segmentation, I discovered distinct behavioral patterns:

1 PM – 5 PM (High Volume, Low Severity): This period sees the max crashes due to "School Run + Work Traffic." Congestion keeps speeds low, resulting in thousands of minor fender-benders (Severity 2).

7 AM – 9 AM (High Severity): While volume is lower than afternoon, severity spikes. Commuters rush to work on highways that aren't fully gridlocked yet, often driving into the sun (East-facing glare), leading to dangerous high-speed collisions.

Actionable Insight: City Planners should focus Medical Response units in the Morning (for injuries) and Tow Trucks in the Afternoon (for clearing blocked lanes).

3. The "Late Night" Surprise

The Data: Accident frequency drops to its lowest point between 12 AM – 4 AM.

The Risk: Despite empty roads, the Severity Rate remains surprisingly high compared to the daily average. This suggests that while accidents are rare at night, they are often catastrophic due to speeding and impaired driving.

The Dashboard: This justified the "Night Owl" filter, allowing police to identify high-risk zones (Nightclubs/Bars) that are statistically invisible during the day.

4. The "Ghostbuster" Deduplication

The Problem: Raw government data often contains duplicate reports from multiple agencies (e.g., State Police and City Police reporting the same crash).

The Solution: I engineered a strict deduplication pipeline to remove records sharing the exact same Latitude, Longitude, and Time.

The Result: This prevented the model from "hallucinating" false density hotspots based on clerical errors, ensuring every cluster represents unique, real-world structural failures.

📸 Visual Analysis (Before vs. After)

1. The Problem: Raw Density Heatmap

File: reports/01_raw_density_heatmap.html (Download repo to view interactively)

A standard heatmap shows "where cars exist," not necessarily where roads are broken. It is messy, overlapping, and unactionable for engineers.

2. The Solution: DBSCAN Centroid Map

File: reports/02_final_cluster_map.html (Download repo to view interactively)

By calculating the Centroid of each cluster, the model produces a "Target Map." Each red dot represents a specific intersection with >15 accidents.

Red Size: Proportional to Severity.

Location: Pinpoint accuracy (65m radius).

🧠 Deep Dive: Parameter Tuning (The Math)

To ensure scientific accuracy, I avoided guessing the clustering parameters. See 1_Analysis_and_Modeling.ipynb for the full code.

K-Distance Graph: I calculated the distance to the 15th nearest neighbor for all points using the Haversine metric.

The "Elbow" Point: The graph showed a sharp inflection point (vertical rise) at approximately 65 meters.

Conclusion: This physical distance corresponds roughly to the size of a standard Miami city intersection. Setting $\epsilon = 65m$ ensured that the model grouped accidents occurring at the same junction while rejecting unconnected crashes 100m away.

🛠️ Tech Stack & Methodology

Algorithm: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Metric: Haversine Distance (Geodesic math for spherical coordinates)

App Framework: Streamlit & PyDeck (Interactive 3D Mapping)

Data Processing: Pandas & NumPy (Vectorized coordinate conversion)

📊 Project Structure

Traffic-Hotspot-Intelligence/
│
├── 📄 app.py                        # The Interactive Streamlit Dashboard
├── 📄 1_Analysis_and_Modeling.ipynb # EDA, K-Distance Graph & Math Verification
├── 📄 requirements.txt              # Dependencies
├── 📄 Miami_Accidents_Cleaned.csv   # Processed Data (Derived from US Accidents Source)
│
├── 📂 reports/
│   ├── 📄 01_raw_density_heatmap.html   # BEFORE: The messy raw data
│   └── 📄 02_final_cluster_map.html     # AFTER: The precise cluster analysis
│
└── 📄 README.md                     # Project Documentation


🚀 How to Run the Dashboard

To interact with the filters (Rush Hour vs. Night Owl) and see the Top 5 Leaderboard:

Clone the repository

git clone [https://github.com/your-username/Traffic-Hotspot-Intelligence.git](https://github.com/your-username/Traffic-Hotspot-Intelligence.git)
cd Traffic-Hotspot-Intelligence


Install Dependencies

pip install -r requirements.txt


Launch the App

streamlit run app.py


📉 Data Source

Original Source: US Accidents (2016 - 2023) on Kaggle

Processing: The original 3GB dataset was filtered to extract only records for Miami, FL. The data was then cleaned to remove duplicates, drop invalid coordinates, and impute missing street names.
