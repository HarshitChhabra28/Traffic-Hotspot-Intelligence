# 🚦 Miami Urban Traffic Intelligence System

## 🔍 Project Overview

**Goal:** Optimize city resource allocation by identifying high-risk traffic zones using **Unsupervised Machine Learning**.

City planners often struggle to differentiate between random *human-error accidents* and *structural road failures*.  
This project applies **DBSCAN clustering** on geospatial accident data (derived from the **US Accidents dataset**) to:

- Remove noise and duplicate records  
- Detect true accident hotspots  
- Pinpoint specific intersections (**"Bullet Holes"**) that require safety intervention  

---

## 💡 Key Intelligence & Results

### 🔴 The Severity Paradox (Core Insight)

> **Accident volume ≠ actual danger**

**Findings**
- **Palmetto Expressway**
  - Accidents: **12,510**
  - Major accidents (Severity 3+): **6%**
- **I-95 South**
  - Major accidents (Severity 3+): **37%**

**Insight**
- High frequency does **not** mean high risk  
- City resources should be prioritized by **severity**, not raw counts  

---

### 🌅 Morning vs. Afternoon Crash Patterns

Distinct temporal behavior emerges when segmenting accidents by time.

#### 🕐 1 PM – 5 PM (High Volume, Low Severity)
- School traffic + office rush  
- Congestion keeps speeds low  
- Thousands of minor collisions (**Severity 2**)  

#### 🌄 7 AM – 9 AM (Lower Volume, High Severity)
- Faster highway speeds  
- East-facing sun glare  
- Higher-impact collisions (**Severity 3+**)  

**✅ Actionable Insight**
- **Morning:** Prioritize **medical response units**  
- **Afternoon:** Deploy **tow trucks** to clear lanes quickly  

---

### 🌙 The Late-Night Risk Factor

**Data**
- Lowest accident frequency: **12 AM – 4 AM**

**Risk**
- Severity rate remains **higher than daily average**
- Likely causes:
  - Speeding  
  - Impaired driving  

**Dashboard Feature**
- **"Night Owl" filter** highlights:
  - Nightclub zones  
  - Bar-dense areas  
  - High-risk intersections invisible during daytime analysis  

---

### 👻 The Ghostbuster Deduplication Strategy

**Problem**  
Government datasets often contain **duplicate accident reports** from multiple agencies.

**Solution**  
A strict deduplication pipeline removes records with identical:
- Latitude  
- Longitude  
- Timestamp  

**Result**
- Prevents false density clusters  
- Ensures every hotspot represents a **real structural issue**  

---

## 📸 Visual Analysis (Before vs. After)

### ❌ Raw Density Heatmap
**File:** `reports/01_raw_density_heatmap.html`  
*Download the repo to view interactively*

- Overlapping noise  
- Shows *where cars exist*  
- Not actionable for engineers  

---

### ✅ DBSCAN Cluster Centroid Map
**File:** `reports/02_final_cluster_map.html`  
*Download the repo to view interactively*

Each red dot represents a **true accident hotspot**.

- 🔴 **Dot size:** Proportional to severity  
- 📍 **Location:** ~65m radius (intersection-level accuracy)  
- 🚦 **Threshold:** >15 accidents per cluster  

---

## 🧠 Deep Dive: Parameter Tuning (The Math)

Clustering parameters were **data-driven**, not guessed.

**Notebook:** `1_Analysis_and_Modeling.ipynb`

### 📈 K-Distance Graph
- Distance to the **15th nearest neighbor**
- Metric: **Haversine (geodesic distance)**

### 🔑 Elbow Detection
- Sharp inflection at **~65 meters**

**Conclusion**
- ~65m ≈ typical Miami intersection  
- Setting `ε = 65m`:
  - Groups accidents at the same junction  
  - Rejects unrelated crashes ~100m away  

---

## 🛠️ Tech Stack & Methodology

- **Algorithm:** DBSCAN  
- **Distance Metric:** Haversine  
- **Visualization:** Streamlit, PyDeck  
- **Data Processing:** Pandas, NumPy  

---

## 📊 Project Structure

```text
Traffic-Hotspot-Intelligence/
│
├── app.py
├── 1_Analysis_and_Modeling.ipynb
├── requirements.txt
├── Miami_Accidents_Cleaned.csv
│
├── reports/
│   ├── 01_raw_density_heatmap.html
│   └── 02_final_cluster_map.html
│
└── README.md

