"""
================================================================================
FEDPULSE AI: Federated National Health Resource & Supply Chain Resilience Platform
================================================================================
Hackathon Track: Smart Health & Supply Chain Resilience

INSTRUCTIONS TO INSTALL AND RUN:
--------------------------------
1. Ensure Python 3.9+ is installed.
2. Install required dependencies:
   pip install streamlit pandas scikit-learn plotly matplotlib numpy

3. Run the Streamlit web application:
   streamlit run app.py
================================================================================
"""

import os
import json
import time
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# ==============================================================================
# 0. STREAMLIT PAGE CONFIG & CUSTOM THEMING (DARK COMMAND CENTER AESTHETICS)
# ==============================================================================
st.set_page_config(
    page_title="FedPulse AI - National Health Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark glassmorphic command center aesthetic
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        margin: 4px 0 0 0;
        font-size: 1.0rem;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    
    /* Section Headers */
    .section-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge-critical {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fde68a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-healthy {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Custom Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "health_resilience.db"

# ==============================================================================
# 1. LOCAL DATABASE ENGINE & SYNTHETIC SEEDING (SQLITE)
# ==============================================================================
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite tables and seed realistic synthetic data for developing nations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. PHC Facilities Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS phc_facilities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        district TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        total_beds INTEGER NOT NULL,
        occupied_beds INTEGER NOT NULL,
        total_staff INTEGER NOT NULL,
        staff_present INTEGER NOT NULL,
        network_status TEXT NOT NULL,
        last_sync TEXT NOT NULL
    )
    """)
    
    # 2. Inventory Stocks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phc_id TEXT NOT NULL,
        medicine_name TEXT NOT NULL,
        category TEXT NOT NULL,
        current_stock INTEGER NOT NULL,
        min_threshold INTEGER NOT NULL,
        daily_burn_rate REAL NOT NULL,
        FOREIGN KEY(phc_id) REFERENCES phc_facilities(id)
    )
    """)

    # 3. Road Logistics & Transit Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS road_logistics (
        district TEXT PRIMARY KEY,
        weather_condition TEXT NOT NULL,
        road_blockage_pct REAL NOT NULL,
        avg_truck_delay_hrs REAL NOT NULL,
        road_infra_score REAL NOT NULL
    )
    """)

    # 4. Satellite Crop Failure & NDVI Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS satellite_ndvi (
        district TEXT PRIMARY KEY,
        ndvi_drop_pct REAL NOT NULL,
        crop_failure_index REAL NOT NULL,
        drought_duration_weeks INTEGER NOT NULL,
        last_satellite_pass TEXT NOT NULL
    )
    """)

    # 5. Offline DTN Store-and-Forward Queue
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dtn_pending_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phc_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        carrier_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)

    conn.commit()

    # Check if DB is already populated
    cursor.execute("SELECT COUNT(*) FROM phc_facilities")
    if cursor.fetchone()[0] == 0:
        seed_synthetic_db_data(conn)
    
    conn.close()

def seed_synthetic_db_data(conn):
    """Seed initial data simulating 12 PHCs across 4 developing rural districts."""
    cursor = conn.cursor()

    # Districts in regional health network
    # Coordinates centered around a developing rural corridor
    phcs = [
        ("PHC-101", "Kagera Central PHC", "Highland District", -1.330, 31.810, 45, 39, 14, 12, "Online", "2026-09-30 08:30"),
        ("PHC-102", "Misenyi River PHC", "Highland District", -1.250, 31.650, 30, 14, 10, 9, "Online", "2026-09-30 08:45"),
        ("PHC-103", "Bukoba Rural PHC", "Highland District", -1.410, 31.780, 50, 48, 16, 14, "Online", "2026-09-30 09:00"),

        ("PHC-201", "Rift Valley North PHC", "Arid Northern Basin", -2.150, 32.400, 35, 33, 12, 6, "Offline-DTN", "2026-09-28 14:10"),
        ("PHC-202", "Dryland Border PHC", "Arid Northern Basin", -2.300, 32.650, 25, 24, 8, 4, "Offline-DTN", "2026-09-27 11:20"),
        ("PHC-203", "Nomad Settlement PHC", "Arid Northern Basin", -2.450, 32.900, 20, 19, 6, 5, "Offline-DTN", "2026-09-26 16:45"),

        ("PHC-301", "River Delta Main PHC", "River Delta District", -1.850, 31.200, 60, 42, 20, 18, "Online", "2026-09-30 09:15"),
        ("PHC-302", "Floodplain Outpost PHC", "River Delta District", -1.950, 31.050, 25, 23, 8, 7, "Online", "2026-09-30 07:50"),
        ("PHC-303", "Lakeside Clinic PHC", "River Delta District", -1.750, 31.350, 40, 31, 12, 11, "Online", "2026-09-30 09:05"),

        ("PHC-401", "Southern Valley PHC", "Southern Valley", -2.750, 31.500, 40, 22, 12, 10, "Online", "2026-09-30 08:00"),
        ("PHC-402", "Forest Margin PHC", "Southern Valley", -2.850, 31.700, 30, 12, 8, 7, "Online", "2026-09-30 08:20"),
        ("PHC-403", "Highland Foothills PHC", "Southern Valley", -2.650, 31.300, 35, 18, 10, 9, "Online", "2026-09-30 08:50"),
    ]
    cursor.executemany("INSERT INTO phc_facilities VALUES (?,?,?,?,?,?,?,?,?,?,?)", phcs)

    # Essential Medical Supplies Inventory per PHC
    medicines = [
        ("Amoxicillin 500mg", "Antibiotics"),
        ("Coartem (Antimalarial)", "Antimalarial"),
        ("Pentavalent Vaccine", "Vaccines"),
        ("Oral Rehydration Salts (ORS)", "Essential"),
        ("Therapeutic Food (RUTF)", "Nutrition")
    ]

    np.random.seed(42)
    inventory_rows = []
    for phc in phcs:
        phc_id = phc[0]
        district = phc[2]
        
        for med_name, category in medicines:
            # Create synthetic shortage condition for Arid Northern Basin & River Delta
            if district == "Arid Northern Basin" and med_name in ["Coartem (Antimalarial)", "Therapeutic Food (RUTF)"]:
                curr_stock = int(np.random.randint(5, 30))
                min_thresh = 150
            elif district == "River Delta District" and med_name == "Amoxicillin 500mg":
                curr_stock = int(np.random.randint(20, 50))
                min_thresh = 200
            elif district == "Highland District":
                curr_stock = int(np.random.randint(400, 900)) # Surplus district
                min_thresh = 200
            else:
                curr_stock = int(np.random.randint(120, 400))
                min_thresh = 100
                
            burn_rate = round(float(np.random.uniform(8.0, 25.0)), 1)
            inventory_rows.append((phc_id, med_name, category, curr_stock, min_thresh, burn_rate))

    cursor.executemany("""
    INSERT INTO inventory_stocks (phc_id, medicine_name, category, current_stock, min_threshold, daily_burn_rate)
    VALUES (?,?,?,?,?,?)
    """, inventory_rows)

    # Road Logistics Conditions
    road_data = [
        ("Highland District", "Clear", 10.0, 1.5, 8.5),
        ("Arid Northern Basin", "Severe Duststorm", 65.0, 14.0, 3.2),
        ("River Delta District", "Monsoon Flooding", 80.0, 22.0, 4.0),
        ("Southern Valley", "Light Rain", 25.0, 4.0, 7.0),
    ]
    cursor.executemany("INSERT INTO road_logistics VALUES (?,?,?,?,?)", road_data)

    # Satellite NDVI & Crop Failure Data
    ndvi_data = [
        ("Highland District", 5.2, 0.12, 1, "2026-09-29"),
        ("Arid Northern Basin", 44.8, 0.82, 10, "2026-09-29"), # Severe crop failure drought
        ("River Delta District", 18.4, 0.35, 3, "2026-09-29"),
        ("Southern Valley", 8.0, 0.18, 2, "2026-09-29"),
    ]
    cursor.executemany("INSERT INTO satellite_ndvi VALUES (?,?,?,?,?)", ndvi_data)

    conn.commit()

init_db()

# ==============================================================================
# 2. SYNTHETIC MACHINE LEARNING ENGINE GENERATORS
# ==============================================================================

@st.cache_resource
def train_transit_supply_chain_model():
    """
    Train a Scikit-Learn Random Forest model that models supply chain transit delays
    and forecasts stockout probability based on road disruptions & truck delays.
    """
    np.random.seed(101)
    N = 1200
    
    # Feature Generation
    daily_burn = np.random.uniform(5, 45, N)
    current_stock = np.random.uniform(10, 600, N)
    nominal_lead_days = np.random.uniform(2, 7, N)
    road_blockage_pct = np.random.uniform(0, 95, N)
    weather_severity = np.random.uniform(0, 10, N) # 0=Clear, 10=Catastrophic
    truck_delay_hrs = np.random.uniform(0, 48, N)
    distance_km = np.random.uniform(25, 350, N)
    
    # Effective lead time formula incorporating real-world transit friction
    transit_delay_days = (truck_delay_hrs / 24.0) + (road_blockage_pct / 100.0) * 4.0 + (weather_severity / 10.0) * 3.0
    total_effective_lead_days = nominal_lead_days + transit_delay_days
    
    expected_consumption_during_transit = daily_burn * total_effective_lead_days
    stockout_margin = current_stock - expected_consumption_during_transit
    
    # Targets
    stockout_occurred = (stockout_margin <= 0).astype(int)
    days_to_stockout = np.maximum(0, current_stock / np.maximum(0.1, daily_burn) - transit_delay_days)
    
    X = pd.DataFrame({
        'daily_burn_rate': daily_burn,
        'current_stock': current_stock,
        'nominal_lead_days': nominal_lead_days,
        'road_blockage_pct': road_blockage_pct,
        'weather_severity': weather_severity,
        'truck_delay_hrs': truck_delay_hrs,
        'distance_km': distance_km
    })
    
    # Train Classification & Regression Models
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, stockout_occurred)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, days_to_stockout)
    
    return clf, reg, X.columns.tolist()


@st.cache_resource
def train_satellite_malnutrition_model():
    """
    Train a Scikit-Learn Gradient Boosting model that maps satellite NDVI drop metrics
    and crop failure indices to expected spikes in pediatric malnutrition visits.
    """
    np.random.seed(202)
    N = 800
    
    ndvi_drop_pct = np.random.uniform(0, 60, N)
    crop_failure_index = np.random.uniform(0.0, 1.0, N)
    drought_weeks = np.random.uniform(0, 20, N)
    baseline_malnutrition_cases = np.random.uniform(10, 80, N)
    temperature_anomaly = np.random.uniform(-1.0, 4.5, N)
    
    # Ground truth non-linear relationship (crop failure leads to delayed malnutrition visit spikes)
    visit_spike_pct = (
        (ndvi_drop_pct * 0.8) + 
        (crop_failure_index * 45.0) + 
        (drought_weeks * 2.2) + 
        (temperature_anomaly * 4.0) + 
        np.random.normal(0, 4.0, N)
    )
    visit_spike_pct = np.clip(visit_spike_pct, 0, 250) # Max 250% spike
    
    X = pd.DataFrame({
        'ndvi_drop_pct': ndvi_drop_pct,
        'crop_failure_index': crop_failure_index,
        'drought_weeks': drought_weeks,
        'baseline_malnutrition_cases': baseline_malnutrition_cases,
        'temperature_anomaly': temperature_anomaly
    })
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, visit_spike_pct)
    
    return model, X.columns.tolist()

# Load Models into Session
clf_transit, reg_transit, transit_feature_names = train_transit_supply_chain_model()
model_sat, sat_feature_names = train_satellite_malnutrition_model()


# ==============================================================================
# 3. SIDEBAR NAVIGATION & HEADER
# ==============================================================================

# App Title & Subtitle
st.markdown("""
<div class="main-header">
    <h1>🛡️ FedPulse AI Command Center</h1>
    <p>National-Scale Federated Health Resource Management & Supply Chain Resilience System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=70)
st.sidebar.title("Navigation Hub")
module_choice = st.sidebar.radio(
    "Select System Module:",
    [
        "🚨 1. Real-Time PHC Command Center",
        "🚚 2. Transit-Time ML Engine",
        "🛰️ 3. Satellite Malnutrition AI",
        "📶 4. DTN Offline Sync Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Network Status")
st.sidebar.markdown("🟢 **Central Hub:** Operational")
st.sidebar.markdown("📡 **Federated Nodes:** 12 Active PHCs")
st.sidebar.markdown("📦 **Pending DTN Packages:** Check Module 4")


# ==============================================================================
# MODULE 1: REAL-TIME PHC COMMAND CENTER (DASHBOARD)
# ==============================================================================
if module_choice == "🚨 1. Real-Time PHC Command Center":
    st.subheader("🚨 Real-Time Health & Resource Command Center")
    st.markdown("Live monitoring of bed capacity, medical staff, inventory levels, and automated cross-district supply redistribution.")
    
    conn = get_db_connection()
    phc_df = pd.read_sql_query("SELECT * FROM phc_facilities", conn)
    inv_df = pd.read_sql_query("""
        SELECT i.*, p.name as phc_name, p.district, p.network_status 
        FROM inventory_stocks i 
        JOIN phc_facilities p ON i.phc_id = p.id
    """, conn)
    conn.close()

    # --- TOP METRIC BANNER ---
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    
    total_phcs = len(phc_df)
    online_phcs = len(phc_df[phc_df['network_status'] == 'Online'])
    total_beds = phc_df['total_beds'].sum()
    occ_beds = phc_df['occupied_beds'].sum()
    bed_rate = round((occ_beds / total_beds) * 100, 1)
    
    total_staff = phc_df['total_staff'].sum()
    present_staff = phc_df['staff_present'].sum()
    staff_rate = round((present_staff / total_staff) * 100, 1)
    
    # Calculate critical inventory items (stock < threshold)
    critical_items = len(inv_df[inv_df['current_stock'] < inv_df['min_threshold']])

    mcol1.metric("Total PHC Facilities", f"{total_phcs}", f"{online_phcs} Connected Online")
    mcol2.metric("Network Bed Occupancy", f"{occ_beds} / {total_beds}", f"{bed_rate}% Full")
    mcol3.metric("Staff Attendance", f"{present_staff} / {total_staff}", f"{staff_rate}% Present")
    mcol4.metric("Critical Stockout Alerts", f"{critical_items} Medicines", f"Requires Action", delta_color="inverse")
    mcol5.metric("Sync Status", "Active", "Delay-Tolerant Mesh")

    st.markdown("---")

    # --- DISTRICT FILTER & MAP ---
    col_filter, col_map = st.columns([1, 2])

    with col_filter:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Command Controls")
        
        selected_district = st.selectbox(
            "Filter Region / District:",
            ["All Districts"] + list(phc_df['district'].unique())
        )
        
        selected_med_cat = st.selectbox(
            "Filter Medicine Category:",
            ["All Categories"] + list(inv_df['category'].unique())
        )
        
        st.markdown("#### Status Legend")
        st.markdown("<span class='badge-healthy'>Healthy PHC</span> Beds < 85%, Stocks Normal", unsafe_allow_html=True)
        st.markdown("<br><span class='badge-warning'>Warning Zone</span> High Bed Load / Low Stock", unsafe_allow_html=True)
        st.markdown("<br><span class='badge-critical'>Critical Zone</span> Critical Shortage / Offline", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Filter Data
    filtered_phc = phc_df if selected_district == "All Districts" else phc_df[phc_df['district'] == selected_district]
    filtered_inv = inv_df if selected_district == "All Districts" else inv_df[inv_df['district'] == selected_district]
    if selected_med_cat != "All Categories":
        filtered_inv = filtered_inv[filtered_inv['category'] == selected_med_cat]

    with col_map:
        st.markdown("### 🗺️ Geospatial Regional Health Warning Map")
        
        # Calculate risk score per PHC for visual coding
        map_df = filtered_phc.copy()
        risk_labels = []
        color_scales = []
        for _, row in map_df.iterrows():
            occ_pct = row['occupied_beds'] / row['total_beds']
            # Check if this PHC has critical stock
            phc_criticals = inv_df[(inv_df['phc_id'] == row['id']) & (inv_df['current_stock'] < inv_df['min_threshold'])]
            
            if len(phc_criticals) > 0 or row['network_status'] == 'Offline-DTN':
                risk_labels.append("CRITICAL RISK")
                color_scales.append("#ef4444")
            elif occ_pct > 0.80:
                risk_labels.append("WARNING")
                color_scales.append("#f59e0b")
            else:
                risk_labels.append("HEALTHY")
                color_scales.append("#10b981")

        map_df['Risk Status'] = risk_labels
        map_df['Marker Color'] = color_scales

        fig_map = px.scatter_map(
            map_df,
            lat="lat",
            lon="lng",
            color="Risk Status",
            color_discrete_map={"CRITICAL RISK": "#ef4444", "WARNING": "#f59e0b", "HEALTHY": "#10b981"},
            size="total_beds",
            hover_name="name",
            hover_data={"district": True, "occupied_beds": True, "total_beds": True, "staff_present": True, "network_status": True},
            zoom=7.5,
            height=380
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # --- INVENTORY & CAPACITY VISUALIZATIONS ---
    st.markdown("---")
    st.markdown("### 📊 Live Stock Levels vs Minimum Safety Thresholds")
    
    # Aggregate stock by medicine
    med_agg = filtered_inv.groupby(['medicine_name', 'phc_name'])[['current_stock', 'min_threshold']].sum().reset_index()
    
    fig_inv = px.bar(
        med_agg,
        x="phc_name",
        y="current_stock",
        color="medicine_name",
        barmode="group",
        title="Current Medicine Reserves Across Selected Facilities (Units)",
        labels={"current_stock": "Units in Stock", "phc_name": "Primary Health Centre"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=400
    )
    fig_inv.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font_color="#e2e8f0"
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    # --- AUTOMATED CROSS-DISTRICT RESOURCE REDISTRIBUTION RECOMMENDER SYSTEM ---
    st.markdown("---")
    st.markdown("### 🔄 Automated Cross-District Resource Redistribution Recommender")
    st.markdown("AI system identifies surplus stocks in neighboring facilities and generates optimal transfer directives to struggling PHCs.")

    # Algorithm to find redistribution opportunities
    # Find deficit items (stock < threshold) and match with surplus items (stock > 2.5 * threshold)
    deficit_items = inv_df[inv_df['current_stock'] < inv_df['min_threshold']].copy()
    surplus_items = inv_df[inv_df['current_stock'] > (inv_df['min_threshold'] * 2.2)].copy()

    recs = []
    for _, def_row in deficit_items.iterrows():
        med = def_row['medicine_name']
        needed = def_row['min_threshold'] - def_row['current_stock'] + 50 # Add buffer
        
        # Look for surplus of the exact same medicine
        matching_surplus = surplus_items[surplus_items['medicine_name'] == med]
        if not matching_surplus.empty:
            # Pick donor with highest surplus
            donor = matching_surplus.sort_values(by='current_stock', ascending=False).iloc[0]
            transfer_qty = min(int((donor['current_stock'] - donor['min_threshold']) * 0.5), int(needed))
            
            if transfer_qty > 10:
                recs.append({
                    "med_name": med,
                    "donor_phc_id": donor['phc_id'],
                    "donor_name": donor['phc_name'],
                    "donor_district": donor['district'],
                    "recipient_phc_id": def_row['phc_id'],
                    "recipient_name": def_row['phc_name'],
                    "recipient_district": def_row['district'],
                    "qty": transfer_qty,
                    "urgency": "CRITICAL" if def_row['current_stock'] < (def_row['min_threshold'] * 0.3) else "HIGH"
                })

    if recs:
        rec_df = pd.DataFrame(recs)
        
        col_rec_list, col_rec_action = st.columns([3, 1])
        with col_rec_list:
            st.dataframe(
                rec_df[['urgency', 'med_name', 'qty', 'donor_name', 'donor_district', 'recipient_name', 'recipient_district']],
                column_config={
                    "urgency": "Priority Level",
                    "med_name": "Medicine",
                    "qty": "Suggested Transfer Qty",
                    "donor_name": "Source Facility (Surplus)",
                    "recipient_name": "Target Facility (Deficit)"
                },
                use_container_width=True,
                hide_index=True
            )
        
        with col_rec_action:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("#### Execute Directive")
            selected_transfer_idx = st.selectbox("Select Recommendation #:", range(len(recs)), format_func=lambda i: f"#{i+1}: {recs[i]['med_name']} ({recs[i]['qty']} units)")
            
            if st.button("🚀 Approve & Dispatch Transfer"):
                t = recs[selected_transfer_idx]
                conn = get_db_connection()
                cur = conn.cursor()
                # Deduct from donor
                cur.execute("UPDATE inventory_stocks SET current_stock = current_stock - ? WHERE phc_id = ? AND medicine_name = ?",
                            (t['qty'], t['donor_phc_id'], t['med_name']))
                # Add to recipient
                cur.execute("UPDATE inventory_stocks SET current_stock = current_stock + ? WHERE phc_id = ? AND medicine_name = ?",
                            (t['qty'], t['recipient_phc_id'], t['med_name']))
                conn.commit()
                conn.close()
                st.success(f"✅ Transfer Directive Executed! Moved {t['qty']} units of {t['med_name']} from {t['donor_name']} to {t['recipient_name']}.")
                time.sleep(1)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("🟢 All network inventory levels are currently balanced above emergency thresholds.")


# ==============================================================================
# MODULE 2: TRANSIT-TIME MACHINE LEARNING ENGINE
# ==============================================================================
elif module_choice == "🚚 2. Transit-Time ML Engine":
    st.subheader("🚚 Transit-Time Machine Learning Engine (Predictive Supply Chain)")
    st.markdown("Advanced Scikit-Learn Predictive Model incorporating real-time road conditions, weather disruptions, and truck delays to forecast stockouts.")

    tab1, tab2 = st.tabs(["🔮 Real-Time Risk Simulation", "📈 ML Model Architecture & Feature Importance"])

    with tab1:
        st.markdown("### 🛠️ Simulate Real-Time Logistics Disruptions")
        
        col_inputs, col_results = st.columns([1, 2])
        
        conn = get_db_connection()
        phc_df = pd.read_sql_query("SELECT * FROM phc_facilities", conn)
        inv_df = pd.read_sql_query("SELECT * FROM inventory_stocks", conn)
        conn.close()
        
        with col_inputs:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("#### Select Target Facility & Road Condition")
            
            target_phc_id = st.selectbox("Target PHC Facility:", phc_df['id'] + " - " + phc_df['name'])
            phc_id_clean = target_phc_id.split(" - ")[0]
            
            selected_phc_inv = inv_df[inv_df['phc_id'] == phc_id_clean]
            target_med = st.selectbox("Select Medicine Stock:", selected_phc_inv['medicine_name'].unique())
            
            curr_stock_val = selected_phc_inv[selected_phc_inv['medicine_name'] == target_med]['current_stock'].values[0]
            burn_rate_val = selected_phc_inv[selected_phc_inv['medicine_name'] == target_med]['daily_burn_rate'].values[0]
            
            st.markdown("---")
            st.markdown("#### Real-Time Transit Variables")
            weather_sev = st.slider("Weather Disruption Severity (0=Clear, 10=Storm)", 0.0, 10.0, 6.5, 0.5)
            road_block = st.slider("Road Infrastructure Blockage %", 0.0, 100.0, 45.0, 5.0)
            truck_delay = st.slider("Delivery Truck Breakaway Delay (Hours)", 0.0, 72.0, 18.0, 2.0)
            lead_days = st.slider("Standard Freight Distance Lead (Days)", 1.0, 10.0, 4.0, 0.5)
            dist_km = st.number_input("Distance from Central Depot (km)", 25, 500, 140)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_results:
            st.markdown("### 🤖 ML Model Stockout Forecast")
            
            # Predict using Random Forest
            sample_features = pd.DataFrame([{
                'daily_burn_rate': burn_rate_val,
                'current_stock': curr_stock_val,
                'nominal_lead_days': lead_days,
                'road_blockage_pct': road_block,
                'weather_severity': weather_sev,
                'truck_delay_hrs': truck_delay,
                'distance_km': dist_km
            }])
            
            prob_stockout = clf_transit.predict_proba(sample_features)[0][1]
            pred_days = reg_transit.predict(sample_features)[0]
            
            # Display Forecast Gauge & Warnings
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.metric("Stockout Risk Probability", f"{round(prob_stockout * 100, 1)}%", 
                          "HIGH RISK" if prob_stockout > 0.5 else "SAFE", delta_color="inverse")
            with rcol2:
                st.metric("Predicted Days Until Depletion", f"{round(pred_days, 1)} Days", 
                          f"Burn Rate: {burn_rate_val} units/day")
                
            if prob_stockout > 0.5:
                st.error(f"🚨 **CRITICAL WARNING:** High probability of stockout for **{target_med}** under current transit disruption! Expected arrival delay will exceed current stock duration.")
                st.markdown("#### Recommended Preemptive Action:")
                st.info(f"⚡ Automatically dispatch express courier from nearest highland depot or initiate local redistribution. Order quantity recommended: **{int(burn_rate_val * (lead_days + 5))} units**.")
            else:
                st.success(f"✅ **SAFE:** Facility has adequate stock reserves to absorb the simulated {truck_delay}h delivery delay.")

            # Comparison plot: Standard Lead Time vs Transit-Adjusted Lead Time
            transit_friction_delay = (truck_delay / 24.0) + (road_block / 100.0) * 4.0 + (weather_sev / 10.0) * 3.0
            total_lead = lead_days + transit_friction_delay
            
            df_comp = pd.DataFrame({
                "Lead Time Metric": ["Standard Lead Time", "Transit-Disrupted Real Lead Time"],
                "Days": [lead_days, total_lead]
            })
            fig_lead = px.bar(df_comp, x="Lead Time Metric", y="Days", color="Lead Time Metric",
                              color_discrete_sequence=["#38bdf8", "#ef4444"], height=280)
            fig_lead.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", font_color="#e2e8f0")
            st.plotly_chart(fig_lead, use_container_width=True)

    with tab2:
        st.markdown("### 🧠 Model Architecture & Feature Importance")
        st.markdown("Demonstrating why traditional static consumption models fail: Road disruptions & truck delays dominate stockout risk.")
        
        # Extract Feature Importances from Classifier
        importances = clf_transit.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': transit_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            feature_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance in Stockout Prediction",
            color="Importance",
            color_continuous_scale="Blues",
            height=350
        )
        fig_imp.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", font_color="#e2e8f0")
        st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# MODULE 3: SATELLITE DATA INTEGRATION (MALNUTRITION PREDICTION)
# ==============================================================================
elif module_choice == "🛰️ 3. Satellite Malnutrition AI":
    st.subheader("🛰️ Satellite Crop Failure Correlation & Malnutrition Early Warning")
    st.markdown("Correlating simulated satellite NDVI (Normalized Difference Vegetation Index) drop metrics with public health clinical visits.")

    conn = get_db_connection()
    sat_df = pd.read_sql_query("SELECT * FROM satellite_ndvi", conn)
    phc_df = pd.read_sql_query("SELECT * FROM phc_facilities", conn)
    conn.close()

    col_sat_controls, col_sat_viz = st.columns([1, 2])

    with col_sat_controls:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 📡 Satellite Data Scanner")
        
        selected_sat_district = st.selectbox("Select Target District:", sat_df['district'].unique())
        dist_sat = sat_df[sat_df['district'] == selected_sat_district].iloc[0]
        
        st.metric("NDVI Vegetation Drop", f"{dist_sat['ndvi_drop_pct']}%", "vs 5-Yr Baseline", delta_color="inverse")
        st.metric("Crop Failure Index", f"{dist_sat['crop_failure_index']} / 1.0", "Severe Anomaly" if dist_sat['crop_failure_index'] > 0.5 else "Normal")
        st.metric("Drought Duration", f"{dist_sat['drought_duration_weeks']} Weeks", "Persistent Dry Spell")
        
        st.markdown("---")
        st.markdown("#### Run AI Malnutrition Forecast")
        temp_anomaly = st.slider("Surface Temp Anomaly (°C above avg)", 0.0, 5.0, 2.8, 0.2)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sat_viz:
        st.markdown(f"### 🔮 60-Day Malnutrition Spike Forecast: **{selected_sat_district}**")
        
        # Run Gradient Boosting Satellite Model
        sat_input = pd.DataFrame([{
            'ndvi_drop_pct': dist_sat['ndvi_drop_pct'],
            'crop_failure_index': dist_sat['crop_failure_index'],
            'drought_weeks': dist_sat['drought_duration_weeks'],
            'baseline_malnutrition_cases': 45.0, # Average baseline
            'temperature_anomaly': temp_anomaly
        }])
        
        predicted_spike = model_sat.predict(sat_input)[0]
        
        scol1, scol2 = st.columns(2)
        with scol1:
            st.metric("Predicted Malnutrition Surge", f"+{round(predicted_spike, 1)}%", "Clinical Admissions", delta_color="inverse")
        with scol2:
            st.metric("Risk Assessment Level", "CRITICAL HIGH" if predicted_spike > 50 else "MODERATE", "Preemptive Action Needed")
            
        # Time-Series Simulation Plot of Satellite Crop Failure vs Lagged Malnutrition Admissions
        weeks = np.arange(1, 13)
        ndvi_trend = np.maximum(0.1, 0.6 - (dist_sat['ndvi_drop_pct']/100.0) * np.sin(weeks/3))
        # Malnutrition visits lag crop failures by ~4-6 weeks
        malnutrition_lagged_visits = 30 + (predicted_spike * 0.8) * (1 / (1 + np.exp(-(weeks - 5))))
        
        ts_df = pd.DataFrame({
            "Week": weeks,
            "Satellite NDVI Vegetation Index": ndvi_trend,
            "Pediatric Malnutrition Admissions": malnutrition_lagged_visits
        })
        
        fig_sat_ts = px.line(
            ts_df, x="Week", y=["Satellite NDVI Vegetation Index", "Pediatric Malnutrition Admissions"],
            title="Satellite NDVI Drop vs 6-Week Lagged Malnutrition Admissions Spike",
            color_discrete_sequence=["#10b981", "#ef4444"],
            height=320
        )
        fig_sat_ts.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", font_color="#e2e8f0")
        st.plotly_chart(fig_sat_ts, use_container_width=True)

        # Actionable Stockpile Preemption
        if predicted_spike > 35:
            st.warning(f"⚠️ **PREEMPTIVE DIRECTIVE:** AI models predict a +{round(predicted_spike, 1)}% spike in severe malnutrition clinical visits over the next 60 days due to local crop failure in {selected_sat_district}.")
            if st.button("📦 Preemptively Stockpile Ready-to-Use Therapeutic Food (RUTF)"):
                conn = get_db_connection()
                cur = conn.cursor()
                # Find PHCs in this district
                dist_phcs = phc_df[phc_df['district'] == selected_sat_district]['id'].tolist()
                for pid in dist_phcs:
                    cur.execute("UPDATE inventory_stocks SET current_stock = current_stock + 350 WHERE phc_id = ? AND medicine_name = 'Therapeutic Food (RUTF)'", (pid,))
                conn.commit()
                conn.close()
                st.success(f"✅ Emergency RUTF Buffer Stock (+350 units each) successfully allocated to all facilities in {selected_sat_district}!")
                time.sleep(1)
                st.rerun()


# ==============================================================================
# MODULE 4: OFFLINE STORE-AND-FORWARD SYNC SIMULATOR (DTN)
# ==============================================================================
elif module_choice == "📶 4. DTN Offline Sync Simulator":
    st.subheader("📶 Delay-Tolerant Network (DTN) Store-and-Forward Simulator")
    st.markdown("Simulating remote rural PHCs logging offline clinical data onto passing transport vehicles (Bluetooth/Wi-Fi Mesh) and syncing to the central cloud once in network range.")

    conn = get_db_connection()
    phc_df = pd.read_sql_query("SELECT * FROM phc_facilities", conn)
    queue_df = pd.read_sql_query("SELECT * FROM dtn_pending_queue", conn)
    conn.close()

    col_offline_form, col_queue_status = st.columns([1, 1.2])

    with col_offline_form:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Rural PHC Offline Data Entry")
        st.caption("Simulates a rural worker recording patient visits & stock usage without cellular network access.")
        
        offline_phc = st.selectbox(
            "Select Offline PHC Facility:",
            phc_df[phc_df['network_status'] == 'Offline-DTN']['id'] + " - " + phc_df[phc_df['network_status'] == 'Offline-DTN']['name']
        )
        selected_phc_id = offline_phc.split(" - ")[0]
        
        patient_count = st.number_input("Today's Patient Footfall:", 1, 200, 38)
        amox_used = st.number_input("Amoxicillin Units Dispensed:", 0, 100, 25)
        coartem_used = st.number_input("Coartem Units Dispensed:", 0, 100, 40)
        rutf_used = st.number_input("RUTF Therapeutic Packets Dispensed:", 0, 100, 15)
        beds_occupied_now = st.number_input("Current Occupied Beds:", 0, 50, 28)
        
        carrier = st.selectbox("Passing Transport Vehicle Carrier:", ["Health Supply Motorcycle #4", "District Vaccine Truck #2", "Community Ambulance #7"])
        
        if st.button("💾 Store Data Packet to Local Vehicle (Offline BLE Transfer)"):
            payload = {
                "phc_id": selected_phc_id,
                "patient_footfall": patient_count,
                "beds_occupied": beds_occupied_now,
                "dispensed": {
                    "Amoxicillin 500mg": amox_used,
                    "Coartem (Antimalarial)": coartem_used,
                    "Therapeutic Food (RUTF)": rutf_used
                }
            }
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO dtn_pending_queue (phc_id, payload_json, carrier_id, created_at, status)
                VALUES (?, ?, ?, ?, 'PENDING')
            """, (selected_phc_id, json.dumps(payload), carrier, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            
            st.success(f"📲 Data packet encrypted & transferred via BLE to **{carrier}**! Stored in pending DTN queue.")
            time.sleep(1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_queue_status:
        st.markdown("### 🚚 Pending DTN Transport Vehicle Queue")
        
        if not queue_df.empty:
            pending_items = queue_df[queue_df['status'] == 'PENDING']
            st.markdown(f"**Pending Packets in Transit:** {len(pending_items)}")
            
            st.dataframe(
                pending_items[['id', 'phc_id', 'carrier_id', 'created_at', 'status']],
                use_container_width=True,
                hide_index=True
            )
            
            if len(pending_items) > 0:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.markdown("#### 📡 City Gateway Sync")
                st.caption("Vehicle has arrived at Bukoba Regional Hub with cellular/fiber connection.")
                
                if st.button("🌐 Sync Vehicle Data to Central Cloud DB"):
                    conn = get_db_connection()
                    cur = conn.cursor()
                    
                    for _, p_row in pending_items.iterrows():
                        p_data = json.loads(p_row['payload_json'])
                        pid = p_data['phc_id']
                        
                        # 1. Update PHC beds & sync time
                        cur.execute("UPDATE phc_facilities SET occupied_beds = ?, last_sync = ? WHERE id = ?",
                                    (p_data['beds_occupied'], datetime.now().strftime("%Y-%m-%d %H:%M"), pid))
                        
                        # 2. Update stock dispensations
                        for med, qty in p_data['dispensed'].items():
                            cur.execute("UPDATE inventory_stocks SET current_stock = MAX(0, current_stock - ?) WHERE phc_id = ? AND medicine_name = ?",
                                        (qty, pid, med))
                            
                        # 3. Mark queue status as SYNCED
                        cur.execute("UPDATE dtn_pending_queue SET status = 'SYNCED' WHERE id = ?", (p_row['id'],))
                        
                    conn.commit()
                    conn.close()
                    
                    st.balloons()
                    st.success("🎉 Federated Cloud DB successfully updated with offline PHC records! Re-triggered AI stockout forecasting engines.")
                    time.sleep(1.5)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No pending DTN vehicle payloads currently in transit.")


# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>FedPulse AI Platform | Smart Health & Supply Chain Resilience Track | Built with Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)
