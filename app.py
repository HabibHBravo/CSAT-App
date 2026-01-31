import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# To run: streamlit run app.py

st.set_page_config(page_title="DNAPL CSAT Tool", layout="wide")

# --- Unit constants ---
LB_TO_KG   = 0.453592
FT3_TO_L   = 28.3168
LBFT3_TO_GML = (LB_TO_KG / FT3_TO_L)     # g/mL per (lb/ft³)

# --- Helpers ---
def to_numeric(series): return pd.to_numeric(series, errors="coerce")
def lbft3_to_gml(x):    return np.nan if pd.isna(x) else x * LBFT3_TO_GML
def geom_mean(x):
    s = pd.to_numeric(x, errors="coerce")
    s = s[(~s.isna()) & (s > 0)]
    return np.exp(np.log(s).mean()) if len(s) else np.nan

# --- Default analytes (editable in UI) ---
ANALYTE_DB = {
    "PCE": {"density_g_ml":1.625,"solubility_mg_L":210.0,"Koc_ml_g":152.0},
    "TCE":   {"density_g_ml":1.460,"solubility_mg_L":1280.0,"Koc_ml_g":94.0},
    "cis-1,2-DCE":             {"density_g_ml":1.284,"solubility_mg_L":460.0,"Koc_ml_g":21.0},
    "Vinyl Chloride":     {"density_g_ml":0.912,"solubility_mg_L":1100.0,"Koc_ml_g":19.0},
    "1,1,1-Trichloroethane":   {"density_g_ml":1.334,"solubility_mg_L":1290.0,"Koc_ml_g":102.0},
    "1,4-Dioxane":             {"density_g_ml":1.033,"solubility_mg_L":0.0,   "Koc_ml_g":17.0},
}

# --- UI: Upload ---
st.title("DNAPL CSAT Tool")
uploaded = st.file_uploader("Upload input file (CSV or Excel)", type=["csv","xlsx","xls"])
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# --- Column mapping ---
st.subheader("Column Mapping")
with st.expander("Map your columns", expanded=True):
    cols = df.columns.tolist()
    col_bulk  = st.selectbox("Bulk density column (lb/ft³)", ["<none>"]+cols, index=(cols.index("Bulk")+1) if "Bulk" in cols else 0)
    col_moist = st.selectbox("Moisture column (%)", ["<none>"]+cols, index=(cols.index("moisture%")+1) if "Moisture" in cols else 0)

out = df.copy()
if col_bulk != "<none>":
    out["bulk_lb_ft3"] = to_numeric(out[col_bulk])
    out["Bulk Density (g/mL)"]   = out["bulk_lb_ft3"].apply(lbft3_to_gml)
else:
    out["bulk_lb_ft3"] = np.nan; out["Bulk Density (g/mL)"] = np.nan

if col_moist != "<none>":
    out["moisture_pCt"] = to_numeric(out[col_moist])
    out["Dry Density (g/mL)"] = out.apply(
        lambda r: r["Bulk Density (g/mL)"]/(1.0 + r["moisture_pCt"]/100.0)
        if pd.notna(r["Bulk Density (g/mL)"]) and pd.notna(r["moisture_pCt"]) else np.nan, axis=1
    )
else:
    out["moisture_pCt"] = np.nan; out["Dry Density (g/mL)"] = np.nan

bulk_valid = out["Bulk Density (g/mL)"].dropna()
bulk_geom = geom_mean(bulk_valid); bulk_avg = bulk_valid.mean() if len(bulk_valid) else np.nan; bulk_med = bulk_valid.median() if len(bulk_valid) else np.nan

# Drop uneeded columns
out.drop(columns=["bulk_lb_ft3", "moisture_pCt"], errors="ignore", inplace=True)

st.subheader("Bulk Density (g/mL) — Summary")
c1,c2,c3 = st.columns(3)
c1.metric("Geometric Mean", f"{bulk_geom:.3f}" if pd.notna(bulk_geom) else "NA")
c2.metric("Average (Mean)", f"{bulk_avg:.3f}" if pd.notna(bulk_avg)  else "NA")
c3.metric("Median",         f"{bulk_med:.3f}" if pd.notna(bulk_med)  else "NA")

# --- Analyte and Site Parameters ---
st.subheader("Parameters")

left, right = st.columns(2)

with left:
    st.markdown("### Analyte Parameters")
    analyte_name = st.selectbox("Analyte", list(ANALYTE_DB.keys()))
    defaults = ANALYTE_DB[analyte_name]
    rho_N = st.number_input("DNAPL density (ρN) [g/mL]", value=float(defaults["density_g_ml"]), step=0.001, format="%.3f")
    Sp    = st.number_input("Average solubility (Sp) [mg/L]", value=float(defaults["solubility_mg_L"]), min_value=0.0, step=1.0, format="%.1f")
    Koc   = st.number_input("Organic carbon - water partition coefficient (Koc) [mL/g]", value=float(defaults["Koc_ml_g"]), min_value=0.0, step=0.1, format="%.1f")

with right:
    st.markdown("### Site Parameters")
    foc   = st.number_input("Fraction organic carbon (foc) [unitless]", value=0.01, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
    phi   = st.number_input("Porosity (φ) [unitless]", value=0.30, min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
    rho_b = st.number_input("Geometric Mean Bulk density (ρb) [g/mL]", value=float(bulk_geom) if pd.notna(bulk_geom) else 1.8, min_value=0.001, step=0.001, format="%.3f")
    Kd   = st.number_input("Soil-water partition coefficient (Kd) [mL/g]", value=Koc * foc, min_value=0.0, step=0.1, format="%.3f")

st.caption(f"**Note:**\n\n"
           f"1. Kd = Koc × foc")

# --- Calculations (DNAPL Saturation and Soil/Porewater partitioning): Cd, Ct_soil, and Ct_pore ---
Ct_soil_mgkg = (Sp / rho_b) * (Kd * rho_b) if rho_b > 0 else np.nan
Ct_soil_mgkg_10perc = Ct_soil_mgkg * 0.1 if pd.notna(Ct_soil_mgkg) else np.nan
Ct_pore_mgkg = (phi * Sp) / rho_b if rho_b > 0 else np.nan
Cd_soil_mgkg = ((0.05 * phi * rho_N * 1_000_000.0)/ rho_b) + Ct_soil_mgkg if rho_b > 0 else np.nan
Cd_soil_gkg = Cd_soil_mgkg / 1000.0 if pd.notna(Cd_soil_mgkg) else np.nan

st.subheader(f"Concentration of {analyte_name} in Soil Corresponding to Threshold DNAPL Saturation")
cAA, cAB = st.columns(2)
cAA.metric("Cd Soil (mg/kg)", f"{Cd_soil_mgkg:.1f}" if pd.notna(Cd_soil_mgkg) else "NA")
cAB.metric("Cd Soil (g/kg)", f"{Cd_soil_gkg:.1f}" if pd.notna(Cd_soil_gkg) else "NA")

st.caption("**Notes:**\n\n"
           "1. Cd Soil = Ct Soil + ((Sr * φ * ρN * 10\u2076)/ ρb)\n\n"
           f"2. Cd Soil = Concentration of {analyte_name} in soil corresponding to threshold DNAPL saturation.\n\n"
           f"3. Ct Soil = Threshold concentration of {analyte_name} in soil based on partitioning relationships.\n\n"
           f"4. Sr = Saturation ratio of {analyte_name} in soil. A value of 0.05 (5%) is used for this calculation.")

st.subheader(f"Threshold Concentration of {analyte_name} in Soil and Pore Fluids Based on Partitioning Relationships")
cA, cB, cC = st.columns(3)
cA.metric("Ct Soil (mg/kg)", f"{Ct_soil_mgkg:.1f}" if pd.notna(Ct_soil_mgkg) else "NA")
cB.metric("Ct Soil 10% (mg/kg)", f"{Ct_soil_mgkg_10perc:.1f}" if pd.notna(Ct_soil_mgkg_10perc) else "NA")
cC.metric("Ct Pore Fluids (mg/kg)", f"{Ct_pore_mgkg:.1f}" if pd.notna(Ct_pore_mgkg) else "NA")

st.caption("**Notes:**\n\n"
           "1. Ct Soil = (Sp / ρb) * (Kd * ρb)\n\n"
           "2. Ct Pore Fluids = (φ * Sp) / ρb")

# --- Hybrid Threshold DNAPL Saturation Range ---
st.subheader("Threshold DNAPL Saturation Range")

use_custom = st.checkbox("Use custom range (instead of default fixed list)", value=False)

# Default fixed list
default_sats = [
    0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31,
    0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,
    0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002,
    0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00009, 0.00008, 0.00007, 0.00006,
    0.00005, 0.00004, 0.00003, 0.00002, 0.00001, 0.000009, 0.000008, 0.000007, 0.000006, 0.000005, 0.000004, 0.000003,
    0.000002, 0.000001, 0.0000009, 0.0000008, 0.0000007, 0.0000006, 0.0000005, 0.0000004, 0.0000003, 0.0000002, 0.0000001
]

if use_custom:
    sat_start = st.number_input("Start Sr", value=0.5, min_value=0.0, max_value=0.9, step=0.01, format="%.2f")
    sat_end   = st.number_input("End Sr",   value=0.0000001, min_value=0.0000001, max_value=0.8, step=0.01, format="%.7f")
    sat_step  = st.number_input("Step Sr",  value=0.01, min_value=0.001, max_value=0.25, step=0.001, format="%.003f")
    sats = np.arange(sat_start, sat_end - np.sign(sat_start - sat_end)*sat_step/2, -sat_step) if sat_start>sat_end \
           else np.arange(sat_start, sat_end + sat_step/2, sat_step)
else:
    sats = default_sats

# --- Build results table ---
rows = []
# Define comments for specific saturation values
sat_comments = {
    0.1: "10% DNAPL Saturation",
    0.05: "Maximum Extent of Mobile DNAPL",
    0.01: "1% DNAPL Saturation",
    0.0004: "Maximum Extent of Residual DNAPL",
    0.000004: "1% CSAT (EPA 1% Rule)"
}

for Sr in sats:
    Cd_mgkg = ((Sr * phi * rho_N * 1_000_000.0)/ rho_b) + ((phi * Kd)/ rho_b) if rho_b > 0 else np.nan
    #Cd_mgkg = ((Sr * phi * rho_N * 1_000_000.0)/ rho_b) + Ct_soil_mgkg if rho_b > 0 else np.nan
    Cd_gkg  = Cd_mgkg / 1000.0 if pd.notna(Cd_mgkg) else np.nan
    Cw_gmL  = (Cd_mgkg * rho_b) / (Kd + phi) if (Kd + phi) > 0 else np.nan
    #Cw_gmL  = (Cd_mgkg * rho_b) / ((Kd * rho_b) + phi) if (Kd + phi) > 0 else np.nan
    Cw_mgL  = min(Cw_gmL, Sp) if (pd.notna(Cw_gmL) and pd.notna(Sp)) else np.nan
    
    # Match comment if saturation matches key (within tolerance)
    comment = ""
    for key, text in sat_comments.items():
        if abs(Sr - key) < 1e-8:  # floating-point tolerance
            comment = text
            break
    
    rows.append({
        "Saturation Range": round(Sr, 7),
        "Cd Soil (mg/kg)": round(Cd_mgkg, 1),
        "Cd Soil (g/kg)": round(Cd_gkg, 4),
        "Cw Groundwater (g/mL)": round(Cw_gmL, 1),
        "Cw Groundwater (mg/L)": round(Cw_mgL, 1),
        "Comment": comment
    })

cd_table = pd.DataFrame(rows)
    
st.subheader("Results Across Saturation Range")

# Display table with wider "Comment" column
st.data_editor(
    cd_table,
    width='content',
    column_config={
        "Comment": st.column_config.TextColumn("Comment", width="medium"),
    },
    disabled=True  # make all fields read-only
)

# --- Plot Section ---
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

st.subheader("DNAPL Saturation and Concentration Distribution Plot")

# Convert table values safely to numeric for plotting
x = pd.to_numeric(cd_table["Cd Soil (mg/kg)"], errors="coerce")
y = pd.to_numeric(cd_table["Cw Groundwater (mg/L)"], errors="coerce")

mask = (~x.isna()) & (~y.isna())
x = x[mask]
y = y[mask]

fig, ax = plt.subplots(figsize=(8, 6))

# --- Main CSAT curve ---
ax.plot(x, y, color="saddlebrown", linewidth=2, label=f"CSAT Curve")

# X-axis log scale (reversed), Y-axis linear
ax.set_xscale("log")
ax.invert_xaxis()

# Axis labels and title
ax.set_xlabel(f"{analyte_name} Soil Concentration (mg/kg)", fontsize=11)
ax.set_ylabel(f"{analyte_name} Groundwater Concentration (mg/L)", fontsize=11)
ax.set_title(f"{analyte_name} Distribution (NAPL Saturation [Sr ≤ 0.5] and Concentration Saturation [Csat])", fontsize=12, pad=12)

# --- Axis formatting ---
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.tick_params(axis="x", which="both", labelsize=9)
ax.tick_params(axis="y", which="both", labelsize=9)

# --- Compute key Cd values ---
def get_cd(sr_val):
    row = cd_table.loc[np.isclose(cd_table["Saturation Range"], sr_val, atol=1e-8)]
    return row["Cd Soil (mg/kg)"].values[0] if not row.empty else None

cd_05 = get_cd(0.05)
cd_0006 = get_cd(0.0006)
cd_0001 = get_cd(0.0001)

y_min, y_max = y.min(), y.max()

# --- Shaded conceptual regions ---
if cd_05:
    ax.axvspan(cd_05, x.max(), color="#8B0000", alpha=0.25, label="Mobile DNAPL")
if cd_0006 and cd_05:
    ax.axvspan(cd_0006, cd_05, color="#FFA500", alpha=0.25, label="Residual/Stable DNAPL")
if cd_0001 and cd_0006:
    ax.axvspan(cd_0001, cd_0006, color="#FFFB00", alpha=0.25, label="Sorbed and Dissolved DNAPL")
if cd_0001:
    ax.axvspan(x.min(), cd_0001, color="#9ACD32", alpha=0.25, label="Dissolved DNAPL")

# --- Reference vertical lines and labels ---
highlight_points = {
    0.05: f"Max Mobile DNAPL (>{cd_05} mg/kg)",
    0.0006: f"Max Residual DNAPL (>{cd_0006} to <{cd_05} mg/kg)",
    0.0001: f"Max Sorbed and Dissolved DNAPL (>{cd_0001} to <{cd_0006} mg/kg)",
    0.000004: "EPA 1% Rule",
    0.0000001: f"Dissolved DNAPL (<{cd_0001} mg/kg)"
}
for sr_val, label in highlight_points.items():
    x_val = get_cd(sr_val)
    if x_val:
        ax.axvline(x_val, linestyle="--", color="gray", linewidth=1)
        ax.text(x_val, y_max * 0.05, label, rotation=90, va="bottom", ha="right", fontsize=8, color="gray")

# --- Set Y range similar to Excel example (optional) ---
ax.set_ylim(0, y_max * 1.05)

# --- Grid, legend, layout ---
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
ax.legend(fontsize=9, loc="upper right", frameon=False)
plt.tight_layout()

st.pyplot(fig)

# --- Download Excel ---
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="xlsxwriter") as xw:
    out.to_excel(xw, index=False, sheet_name="Samples")
    pd.DataFrame({"Metric":["Geometric Mean","Average (Mean)","Median"],
                  "Bulk Density (g/mL)":[bulk_geom, bulk_avg, bulk_med]}).to_excel(xw, index=False, sheet_name="Bulk_Stats")
    pd.DataFrame({
        "Parameter":["Analyte","ρN (g/mL)","Solubility (mg/L)","Koc (mL/g)","foc","Kd (mL/g)","Porosity φ","ρb (g/mL)","Ct_soil (mg/kg)","Ct_pore (mg/kg)"],
        "Value":[analyte_name, rho_N, Sp, Koc, foc, Kd, phi, rho_b, Ct_soil_mgkg, Ct_pore_mgkg],
    }).to_excel(xw, index=False, sheet_name="Parameters")
    cd_table.to_excel(xw, index=False, sheet_name="Thresholds")

    # Adjust column widths in Excel output
    worksheet = xw.sheets["Thresholds"]
    for i, col in enumerate(cd_table.columns, 1):
        if col == "Comment":
            worksheet.set_column(i - 1, i - 1, 40)  # 40-character width for comments
        else:
            worksheet.set_column(i - 1, i - 1, 18)  # narrower for others

buff.seek(0)
st.download_button("Download DNAPL_CSAT_results.xlsx", buff,
                   file_name="DNAPL_CSAT_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption(
    "**Notes:**\n\n"
    "1. Conversion from lb/ft³ to g/mL uses (0.453592/28.3168)\n\n"
    "2. Dry density = bulk density / (1 + moisture_fraction)\n\n"
    "3. Cw Groundwater = (Cd Soil * ρb) / (Kd + φ)\n\n"
)
