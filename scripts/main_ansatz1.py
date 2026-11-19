from pathlib import Path
import pandas as pd
from collections import defaultdict
import math

# =========================================================
# 1) DATEI EINLESEN
# =========================================================

datei = str(Path(__file__).resolve().parent.parent / "data" / "demand_104_setup2.xlsx")

df_demand_all = pd.read_excel(
    datei,
    sheet_name="Nachfrage_104_Wochen",
    header=0
)

# Nur Analysejahr: Wochen 53-104
df_demand = df_demand_all[df_demand_all["Woche"].between(53, 104)].copy()

produkte = ["A", "B", "C", "D"]
perioden = df_demand["Woche"].astype(int).tolist()

# Echte Nachfrage
demand = {}
for _, row in df_demand.iterrows():
    t = int(row["Woche"])
    demand[(t, "A")] = float(row["Produkt A"])
    demand[(t, "B")] = float(row["Produkt B"])
    demand[(t, "C")] = float(row["Produkt C"])
    demand[(t, "D")] = float(row["Produkt D"])

# Historie für Forecaststart: Wochen 1-52
df_history = df_demand_all[df_demand_all["Woche"].between(1, 52)].copy()

# =========================================================
# 2) PARAMETER
# =========================================================

kapazitaet = 260
alpha = 0.3

batch = {
    "A": 40,
    "B": 60,
    "C": 20,
    "D": 30
}

haltbarkeit = {
    "A": 6,
    "B": 4,
    "C": 2,
    "D": 3
}

anfangsbestand = {
    "A": 120,
    "B": 80,
    "C": 40,
    "D": 60
}

# (s,S)-Politik
s_param = {
    "A": 120,
    "B": 100,
    "C": 20,
    "D": 80
}

S_param = {
    "A": 240,
    "B": 180,
    "C": 40,
    "D": 140
}

# =========================================================
# 3) FORECAST MIT EXPONENTIELLER GLÄTTUNG
#    Forecast für Analysejahr auf Basis der Historie erzeugen
# =========================================================

forecast = {}

for produkt in produkte:
    col = f"Produkt {produkt}"
    historie = df_history[col].astype(float).tolist()
    analyse = df_demand[col].astype(float).tolist()

    # Letzter geglätteter Wert aus Historie
    f_prev = historie[0]
    for t in range(1, len(historie)):
        f_prev = alpha * historie[t - 1] + (1 - alpha) * f_prev

    # Forecasts für Analysejahr
    # Woche 53:
    forecast[(53, produkt)] = f_prev

    # Ab Woche 54 rollierend mit realisierter Nachfrage aus Vorperiode
    analyse_wochen = df_demand["Woche"].astype(int).tolist()
    for idx in range(1, len(analyse_wochen)):
        t = analyse_wochen[idx]
        t_prev = analyse_wochen[idx - 1]
        forecast[(t, produkt)] = alpha * demand[(t_prev, produkt)] + (1 - alpha) * forecast[(t_prev, produkt)]

# =========================================================
# 4) SIMULATION ANSATZ 1
#    - (s,S)-Politik
#    - Prioritätsregel: niedrigste Deckungsreichweite
#    - Vorlaufzeit: 1 Woche
# =========================================================

# Altersstruktur Lager
inventory_age = {}
for i in produkte:
    L = haltbarkeit[i]
    inventory_age[i] = [0] * L
    inventory_age[i][0] = anfangsbestand[i]

# Pipeline wegen Vorlaufzeit = 1
pipeline = {(t, i): 0 for t in range(1, max(perioden) + 2) for i in produkte}

# KPI-Speicher
fulfilled = defaultdict(float)
total_demand = defaultdict(float)
total_waste = defaultdict(float)
sum_end_inventory = defaultdict(float)
sum_production = defaultdict(float)
sum_shortage = defaultdict(float)
weekly_capacity_usage = []

detail_rows = []

for t in perioden:
    # -----------------------------------------
    # 1) Produktion aus Vorperiode kommt an
    # -----------------------------------------
    for i in produkte:
        ankunft = pipeline[(t, i)]
        if ankunft > 0:
            inventory_age[i][0] += ankunft

    # -----------------------------------------
    # 2) Nachfrage bedienen (älteste zuerst)
    # -----------------------------------------
    shortage_now = {}
    end_inventory_before_prod = {}
    range_now = {}
    reorder_need = {}

    for i in produkte:
        L = haltbarkeit[i]
        real_demand = demand[(t, i)]
        total_demand[i] += real_demand

        remaining = real_demand

        # FEFO / älteste zuerst
        for age_idx in range(L - 1, -1, -1):
            usable = min(inventory_age[i][age_idx], remaining)
            inventory_age[i][age_idx] -= usable
            remaining -= usable

        served = real_demand - remaining
        shortage = remaining

        fulfilled[i] += served
        sum_shortage[i] += shortage
        shortage_now[i] = shortage

        end_inv = sum(inventory_age[i])
        end_inventory_before_prod[i] = end_inv

        # Deckungsreichweite
        f = forecast[(t, i)]
        if f > 0:
            range_now[i] = end_inv / f
        else:
            range_now[i] = 999999

        # (s,S)-Politik
        if end_inv < s_param[i]:
            reorder_need[i] = max(0, S_param[i] - end_inv)
        else:
            reorder_need[i] = 0

    # -----------------------------------------
    # 3) Priorisierung der Produktion
    #    niedrigste Deckungsreichweite zuerst
    # -----------------------------------------
    kandidaten = [i for i in produkte if reorder_need[i] > 0]
    kandidaten = sorted(kandidaten, key=lambda x: range_now[x])

    remaining_capacity = kapazitaet
    produced_this_week = {i: 0 for i in produkte}

    for i in kandidaten:
        if remaining_capacity < batch[i]:
            continue

        # Benötigte Anzahl Batches
        need = reorder_need[i]
        batches_needed = math.ceil(need / batch[i])

        produced = 0
        for _ in range(batches_needed):
            if remaining_capacity >= batch[i]:
                produced += batch[i]
                remaining_capacity -= batch[i]
            else:
                break

        produced_this_week[i] = produced
        sum_production[i] += produced

    weekly_capacity_usage.append((kapazitaet - remaining_capacity) / kapazitaet)

    # -----------------------------------------
    # 4) Produktion in Pipeline für nächste Woche
    # -----------------------------------------
    for i in produkte:
        if t + 1 <= max(perioden):
            pipeline[(t + 1, i)] = produced_this_week[i]

    # -----------------------------------------
    # 5) Verfall am Periodenende
    # -----------------------------------------
    waste_now_dict = {}

    for i in produkte:
        L = haltbarkeit[i]

        # Älteste Restmenge verfällt
        waste_now = inventory_age[i][L - 1]
        total_waste[i] += waste_now
        waste_now_dict[i] = waste_now
        inventory_age[i][L - 1] = 0

    # -----------------------------------------
    # 6) Alterung für nächste Periode
    # -----------------------------------------
    for i in produkte:
        L = haltbarkeit[i]

        for age_idx in range(L - 1, 0, -1):
            inventory_age[i][age_idx] = inventory_age[i][age_idx - 1]

        inventory_age[i][0] = 0  # frische Ware kommt erst nächste Woche aus Pipeline

        end_inventory = sum(inventory_age[i])
        sum_end_inventory[i] += end_inventory

        detail_rows.append({
            "Woche": t,
            "Produkt": i,
            "Echte_Nachfrage": demand[(t, i)],
            "Forecast": round(forecast[(t, i)], 4),
            "Bediente_Nachfrage": demand[(t, i)] - shortage_now[i],
            "Fehlmenge": shortage_now[i],
            "Bestand_nach_Nachfrage": end_inventory_before_prod[i],
            "Deckungsreichweite": round(range_now[i], 4),
            "Nachbestellbedarf": reorder_need[i],
            "Produktion": produced_this_week[i],
            "Verfall": waste_now_dict[i],
            "Endbestand": end_inventory
        })

# =========================================================
# 5) KPI BERECHNEN
# =========================================================

kpi_rows = []

gesamt_demand = sum(total_demand[i] for i in produkte)
gesamt_fulfilled = sum(fulfilled[i] for i in produkte)
gesamt_waste = sum(total_waste[i] for i in produkte)
gesamt_end_inventory = sum(sum_end_inventory[i] for i in produkte)
gesamt_production = sum(sum_production[i] for i in produkte)
gesamt_shortage = sum(sum_shortage[i] for i in produkte)

service_level_gesamt = gesamt_fulfilled / gesamt_demand if gesamt_demand > 0 else 0
stockout_rate_gesamt = gesamt_shortage / gesamt_demand if gesamt_demand > 0 else 0
avg_inventory_gesamt = gesamt_end_inventory / len(perioden)
utilization_gesamt = sum(weekly_capacity_usage) / len(weekly_capacity_usage)

kpi_rows.append({
    "Ebene": "Gesamt",
    "Service_Level": service_level_gesamt,
    "Stockout_Rate": stockout_rate_gesamt,
    "Durchschnittlicher_Bestand": avg_inventory_gesamt,
    "Verfall": gesamt_waste,
    "Produktionsauslastung": utilization_gesamt
})

for i in produkte:
    service_level = fulfilled[i] / total_demand[i] if total_demand[i] > 0 else 0
    stockout_rate = sum_shortage[i] / total_demand[i] if total_demand[i] > 0 else 0
    avg_inventory = sum_end_inventory[i] / len(perioden)
    utilization = sum_production[i] / (kapazitaet * len(perioden))

    kpi_rows.append({
        "Ebene": i,
        "Service_Level": service_level,
        "Stockout_Rate": stockout_rate,
        "Durchschnittlicher_Bestand": avg_inventory,
        "Verfall": total_waste[i],
        "Produktionsauslastung": utilization
    })

# Forecastfehler
forecast_error_rows = []
for i in produkte:
    abs_errors = [abs(demand[(t, i)] - forecast[(t, i)]) for t in perioden]
    sq_errors = [(demand[(t, i)] - forecast[(t, i)]) ** 2 for t in perioden]

    mae = sum(abs_errors) / len(abs_errors)
    mse = sum(sq_errors) / len(sq_errors)

    forecast_error_rows.append({
        "Produkt": i,
        "MAE": mae,
        "MSE": mse
    })

# =========================================================
# 6) ERGEBNISSE AUSGEBEN
# =========================================================

df_results = pd.DataFrame(detail_rows)
df_kpis = pd.DataFrame(kpi_rows)
df_forecast_error = pd.DataFrame(forecast_error_rows)

print("\n==== KPIs ANSATZ 1 ====")
print(df_kpis)

print("\n==== FORECAST-FEHLER ====")
print(df_forecast_error)

print("\n==== ERSTE 10 ZEILEN DETAILERGEBNISSE ====")
print(df_results.head(10))

# =========================================================
# 7) IN EXCEL SPEICHERN
# =========================================================

output_datei = str(Path(__file__).resolve().parent.parent / "results" / "ergebnisse_ansatz1_104.xlsx")

Path(output_datei).parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(output_datei, engine="openpyxl") as writer:
    df_results.to_excel(writer, sheet_name="Detailergebnisse", index=False)
    df_kpis.to_excel(writer, sheet_name="KPIs", index=False)
    df_forecast_error.to_excel(writer, sheet_name="Forecastfehler", index=False)

print(f"\nErgebnisse gespeichert unter:\n{output_datei}")