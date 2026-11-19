from pathlib import Path
import pandas as pd
import pulp

# =========================================================
# 1) DATEI EINLESEN
# =========================================================

datei = str(Path(__file__).resolve().parent.parent / "data" / "demand_104_setup2.xlsx")

df_demand_all = pd.read_excel(
    datei,
    sheet_name="Nachfrage_104_Wochen",
    header=0
)

# Historie: Wochen 1-52
df_history = df_demand_all[df_demand_all["Woche"].between(1, 52)].copy()

# Analysejahr: Wochen 53-104
df_demand = df_demand_all[df_demand_all["Woche"].between(53, 104)].copy()

produkte = ["A", "B", "C", "D"]
perioden = df_demand["Woche"].astype(int).tolist()

# Echte Nachfrage Dictionary
demand = {}

for _, row in df_demand.iterrows():
    t = int(row["Woche"])
    demand[(t, "A")] = float(row["Produkt A"])
    demand[(t, "B")] = float(row["Produkt B"])
    demand[(t, "C")] = float(row["Produkt C"])
    demand[(t, "D")] = float(row["Produkt D"])

# =========================================================
# 2) FORECAST MIT EXPONENTIELLER GLÄTTUNG
#    Historie = Wochen 1-52
#    Analysejahr = Wochen 53-104
# =========================================================

alpha = 0.3
forecast = {}

for produkt in produkte:
    spalte = f"Produkt {produkt}"

    historie = df_history[spalte].astype(float).tolist()
    analyse = df_demand[spalte].astype(float).tolist()

    # Glättungszustand auf Basis der Historie aufbauen
    f_prev = historie[0]
    for idx in range(1, len(historie)):
        f_prev = alpha * historie[idx - 1] + (1 - alpha) * f_prev

    # Forecast für erste Analysewoche
    forecast[(53, produkt)] = float(f_prev)

    # Rollierend für restliche Wochen im Analysejahr
    analyse_wochen = df_demand["Woche"].astype(int).tolist()
    for idx in range(1, len(analyse_wochen)):
        t = analyse_wochen[idx]
        t_prev = analyse_wochen[idx - 1]
        forecast[(t, produkt)] = alpha * demand[(t_prev, produkt)] + (1 - alpha) * forecast[(t_prev, produkt)]

# =========================================================
# 3) PARAMETER
# =========================================================

kapazitaet = 260

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

lagerkosten = {i: 1 for i in produkte}
fehlmengenkosten = {i: 50 for i in produkte}
verfallskosten = {i: 5 for i in produkte}

print("Schritt 1: Daten und Forecast eingelesen")

# =========================================================
# 4) MILP AUFBAUEN
# =========================================================

model = pulp.LpProblem("Ansatz2_MILP_Produktion", pulp.LpMinimize)

# Variablen
# k[t,i] = Anzahl produzierter Batches
k = pulp.LpVariable.dicts(
    "Batches",
    (perioden, produkte),
    lowBound=0,
    cat="Integer"
)

# x[t,i] = produzierte Menge
x = {}
for t in perioden:
    for i in produkte:
        x[(t, i)] = batch[i] * k[t][i]

# Weitere Variablen
serve = {}
carry = {}
waste = {}
short = {}

for t in perioden:
    for i in produkte:
        short[(t, i)] = pulp.LpVariable(f"Short_{t}_{i}", lowBound=0)
        waste[(t, i)] = pulp.LpVariable(f"Waste_{t}_{i}", lowBound=0)

        for a in range(1, haltbarkeit[i] + 1):
            serve[(t, i, a)] = pulp.LpVariable(f"Serve_{t}_{i}_{a}", lowBound=0)
            carry[(t, i, a)] = pulp.LpVariable(f"Carry_{t}_{i}_{a}", lowBound=0)

# Zielfunktion
model += pulp.lpSum(
    lagerkosten[i] * pulp.lpSum(carry[(t, i, a)] for a in range(1, haltbarkeit[i] + 1))
    + fehlmengenkosten[i] * short[(t, i)]
    + verfallskosten[i] * waste[(t, i)]
    for t in perioden for i in produkte
)

# Nebenbedingungen

# (1) Kapazität pro Woche
for t in perioden:
    model += pulp.lpSum(x[(t, i)] for i in produkte) <= kapazitaet

# (2) Nachfrageerfüllung
for t in perioden:
    for i in produkte:
        model += pulp.lpSum(serve[(t, i, a)] for a in range(1, haltbarkeit[i] + 1)) + short[(t, i)] == forecast[(t, i)]

# (3) Bestandslogik
for idx_t, t in enumerate(perioden):
    for i in produkte:
        L = haltbarkeit[i]

        for a in range(1, L + 1):
            # verfügbare Menge bestimmen
            if a == 1:
                if idx_t == 0:
                    available = anfangsbestand[i]
                else:
                    t_prev = perioden[idx_t - 1]
                    available = x[(t_prev, i)]
            else:
                if idx_t == 0:
                    available = 0
                else:
                    t_prev = perioden[idx_t - 1]
                    available = carry[(t_prev, i, a - 1)]

            # Bedienung kann verfügbare Menge nicht überschreiten
            model += serve[(t, i, a)] <= available

            # Trage-/Verfallslogik
            if a < L:
                model += carry[(t, i, a)] == available - serve[(t, i, a)]
            else:
                model += carry[(t, i, a)] == 0
                model += waste[(t, i)] == available - serve[(t, i, a)]

print("Schritt 2: Modell aufgebaut, Solver startet")

# =========================================================
# 5) MODELL LÖSEN
# =========================================================

solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120)
model.solve(solver)

print("Schritt 3: Solver fertig")
print("\n====================================")
print("MILP STATUS:", pulp.LpStatus[model.status])
print("Zielfunktionswert:", round(pulp.value(model.objective), 2))
print("====================================\n")

# =========================================================
# 6) PRODUKTIONSPLAN AUS MILP HOLEN
# =========================================================

production_plan = {}
for t in perioden:
    for i in produkte:
        batches_val = k[t][i].varValue if k[t][i].varValue is not None else 0
        production_plan[(t, i)] = batch[i] * batches_val

# =========================================================
# 7) PLAN MIT ECHTER NACHFRAGE SIMULIEREN
#    Ziel: reale KPIs berechnen
# =========================================================

inventory_age = {}
for i in produkte:
    inventory_age[i] = [0] * haltbarkeit[i]
    inventory_age[i][0] = anfangsbestand[i]

pipeline = {(t, i): 0 for t in range(min(perioden), max(perioden) + 2) for i in produkte}

detail_rows = []

sum_demand = {i: 0 for i in produkte}
sum_fulfilled = {i: 0 for i in produkte}
sum_shortage = {i: 0 for i in produkte}
sum_waste = {i: 0 for i in produkte}
sum_end_inventory = {i: 0 for i in produkte}
sum_production = {i: 0 for i in produkte}
weekly_capacity_used_list = []

for t in perioden:
    # 1) Produktion aus Vorperiode kommt an
    for i in produkte:
        if pipeline[(t, i)] > 0:
            inventory_age[i][0] += pipeline[(t, i)]

    weekly_capacity_used = 0

    for i in produkte:
        L = haltbarkeit[i]
        real_demand = demand[(t, i)]
        sum_demand[i] += real_demand

        # 2) Nachfrage bedienen: FEFO
        remaining = real_demand
        for age_idx in range(L - 1, -1, -1):
            used = min(inventory_age[i][age_idx], remaining)
            inventory_age[i][age_idx] -= used
            remaining -= used

        fulfilled_now = real_demand - remaining
        shortage_now = remaining

        sum_fulfilled[i] += fulfilled_now
        sum_shortage[i] += shortage_now

        # 3) Verfall
        waste_now = inventory_age[i][L - 1]
        sum_waste[i] += waste_now
        inventory_age[i][L - 1] = 0

        # 4) Alterung für nächste Woche
        for age_idx in range(L - 1, 0, -1):
            inventory_age[i][age_idx] = inventory_age[i][age_idx - 1]
        inventory_age[i][0] = 0

        # 5) Endbestand
        end_inventory = sum(inventory_age[i])
        sum_end_inventory[i] += end_inventory

        # 6) Produktion dieser Woche
        prod_now = production_plan[(t, i)]
        sum_production[i] += prod_now
        weekly_capacity_used += prod_now

        if t + 1 <= max(perioden):
            pipeline[(t + 1, i)] = prod_now

        detail_rows.append({
            "Woche": t,
            "Produkt": i,
            "Echte_Nachfrage": real_demand,
            "Forecast": forecast[(t, i)],
            "Produktion": prod_now,
            "Erfuellte_Nachfrage": fulfilled_now,
            "Fehlmenge": shortage_now,
            "Verfall": waste_now,
            "Endbestand": end_inventory
        })

    weekly_capacity_used_list.append(weekly_capacity_used / kapazitaet)

# =========================================================
# 8) KPIs BERECHNEN
# =========================================================

kpi_rows = []

# Gesamt
gesamt_demand = sum(sum_demand.values())
gesamt_fulfilled = sum(sum_fulfilled.values())
gesamt_shortage = sum(sum_shortage.values())
gesamt_waste = sum(sum_waste.values())
gesamt_end_inventory = sum(sum_end_inventory.values())

kpi_rows.append({
    "Ebene": "Gesamt",
    "Service_Level": gesamt_fulfilled / gesamt_demand if gesamt_demand > 0 else 0,
    "Stockout_Rate": gesamt_shortage / gesamt_demand if gesamt_demand > 0 else 0,
    "Durchschnittlicher_Bestand": gesamt_end_inventory / len(perioden),
    "Verfall": gesamt_waste,
    "Produktionsauslastung": sum(weekly_capacity_used_list) / len(weekly_capacity_used_list)
})

# Pro Produkt
for i in produkte:
    kpi_rows.append({
        "Ebene": i,
        "Service_Level": sum_fulfilled[i] / sum_demand[i] if sum_demand[i] > 0 else 0,
        "Stockout_Rate": sum_shortage[i] / sum_demand[i] if sum_demand[i] > 0 else 0,
        "Durchschnittlicher_Bestand": sum_end_inventory[i] / len(perioden),
        "Verfall": sum_waste[i],
        "Produktionsauslastung": sum_production[i] / (kapazitaet * len(perioden))
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
# 9) ERGEBNISSE AUSGEBEN
# =========================================================

df_results = pd.DataFrame(detail_rows)
df_kpis = pd.DataFrame(kpi_rows)
df_forecast_error = pd.DataFrame(forecast_error_rows)

print("\n==== KPIs ANSATZ 2 ====")
print(df_kpis)

print("\n==== FORECAST-FEHLER ====")
print(df_forecast_error)

print("\n==== ERSTE 10 ZEILEN DETAILERGEBNISSE ====")
print(df_results.head(10))

# =========================================================
# 10) IN EXCEL SPEICHERN
# =========================================================

output_datei = str(Path(__file__).resolve().parent.parent / "results" / "ergebnisse_ansatz2_104.xlsx")

Path(output_datei).parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(output_datei, engine="openpyxl") as writer:
    df_results.to_excel(writer, sheet_name="Detailergebnisse", index=False)
    df_kpis.to_excel(writer, sheet_name="KPIs", index=False)
    df_forecast_error.to_excel(writer, sheet_name="Forecastfehler", index=False)

print(f"\nErgebnisse gespeichert unter:\n{output_datei}")