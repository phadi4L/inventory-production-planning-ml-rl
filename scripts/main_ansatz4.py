from pathlib import Path
import math
import pandas as pd
import pulp
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# 1) DATEI EINLESEN
# =========================================================

datei = str(Path(__file__).resolve().parent.parent / "data" / "demand_104_setup2.xlsx")

df_all = pd.read_excel(
    datei,
    sheet_name="Nachfrage_104_Wochen",
    header=0
)

produkte = ["A", "B", "C", "D"]

# Train/Test-Split für Rolling Retraining
# Initiales Training: Wochen 1-52
# Prognose- und Analysezeitraum: Wochen 53-104
df_initial_train = df_all[df_all["Woche"].between(1, 52)].copy()
df_test = df_all[df_all["Woche"].between(53, 104)].copy()

perioden = df_test["Woche"].astype(int).tolist()

# Echte Nachfrage im Testzeitraum
demand = {}
for _, row in df_test.iterrows():
    t = int(row["Woche"])
    demand[(t, "A")] = float(row["Produkt A"])
    demand[(t, "B")] = float(row["Produkt B"])
    demand[(t, "C")] = float(row["Produkt C"])
    demand[(t, "D")] = float(row["Produkt D"])

# =========================================================
# 2) RANDOM-FOREST-FORECAST MIT ROLLING RETRAINING
#    Features:
#    - Lag1 bis Lag4
#    - Zeitindex
#    - sin/cos Saison
#    Vorgehen:
#    - Für jede Prognosewoche t wird das Modell neu auf
#      allen bis t-1 verfügbaren echten Daten trainiert
#    - Danach wird genau Woche t prognostiziert
# =========================================================

forecast = {}
n_lags = 4

for produkt in produkte:
    spalte = f"Produkt {produkt}"

    # Vollständige echte Serie für dieses Produkt
    full_series = df_all[spalte].astype(float).tolist()

    # Rolling Forecast für Wochen 53-104
    for week in perioden:
        train_end_week = week - 1
        serie_train = full_series[:train_end_week]

        X_train = []
        y_train = []

        # Trainingsdaten aus allen bis dahin verfügbaren echten Beobachtungen
        for idx in range(n_lags, len(serie_train)):
            train_week_number = idx + 1  # echte Wochennummer im Training

            lag1 = serie_train[idx - 1]
            lag2 = serie_train[idx - 2]
            lag3 = serie_train[idx - 3]
            lag4 = serie_train[idx - 4]

            sin52 = math.sin((2 * math.pi * train_week_number) / 52)
            cos52 = math.cos((2 * math.pi * train_week_number) / 52)

            X_train.append([lag1, lag2, lag3, lag4, train_week_number, sin52, cos52])
            y_train.append(serie_train[idx])

        model_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42
        )

        model_rf.fit(X_train, y_train)

        # Features für die aktuelle Prognosewoche week
        lag1 = serie_train[-1]
        lag2 = serie_train[-2]
        lag3 = serie_train[-3]
        lag4 = serie_train[-4]

        sin52 = math.sin((2 * math.pi * week) / 52)
        cos52 = math.cos((2 * math.pi * week) / 52)

        x_input = [[lag1, lag2, lag3, lag4, week, sin52, cos52]]
        y_pred = model_rf.predict(x_input)[0]
        y_pred = max(0, float(y_pred))

        forecast[(week, produkt)] = y_pred

print("Schritt 1: RF-Forecast mit Rolling Retraining erstellt")

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

# =========================================================
# 4) MILP AUFBAUEN
# =========================================================

model = pulp.LpProblem("Ansatz3_RF_MILP", pulp.LpMinimize)

k = pulp.LpVariable.dicts(
    "Batches",
    (perioden, produkte),
    lowBound=0,
    cat="Integer"
)

x = {}
for t in perioden:
    for i in produkte:
        x[(t, i)] = batch[i] * k[t][i]

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

model += pulp.lpSum(
    lagerkosten[i] * pulp.lpSum(carry[(t, i, a)] for a in range(1, haltbarkeit[i] + 1))
    + fehlmengenkosten[i] * short[(t, i)]
    + verfallskosten[i] * waste[(t, i)]
    for t in perioden for i in produkte
)

# Kapazität
for t in perioden:
    model += pulp.lpSum(x[(t, i)] for i in produkte) <= kapazitaet

# Nachfrageerfüllung auf Forecast-Basis
for t in perioden:
    for i in produkte:
        model += (
            pulp.lpSum(serve[(t, i, a)] for a in range(1, haltbarkeit[i] + 1))
            + short[(t, i)]
            == forecast[(t, i)]
        )

# Bestandslogik
for idx_t, t in enumerate(perioden):
    for i in produkte:
        L = haltbarkeit[i]

        for a in range(1, L + 1):
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

            model += serve[(t, i, a)] <= available

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
print("MILP STATUS:", pulp.LpStatus[model.status])

# =========================================================
# 6) PRODUKTIONSPLAN HOLEN
# =========================================================

production_plan = {}
for t in perioden:
    for i in produkte:
        batches_val = k[t][i].varValue if k[t][i].varValue is not None else 0
        production_plan[(t, i)] = batch[i] * batches_val

# =========================================================
# 7) MIT ECHTER NACHFRAGE SIMULIEREN
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
    for i in produkte:
        if pipeline[(t, i)] > 0:
            inventory_age[i][0] += pipeline[(t, i)]

    weekly_capacity_used = 0

    for i in produkte:
        L = haltbarkeit[i]
        real_demand = demand[(t, i)]
        sum_demand[i] += real_demand

        remaining = real_demand
        for age_idx in range(L - 1, -1, -1):
            used = min(inventory_age[i][age_idx], remaining)
            inventory_age[i][age_idx] -= used
            remaining -= used

        fulfilled_now = real_demand - remaining
        shortage_now = remaining

        sum_fulfilled[i] += fulfilled_now
        sum_shortage[i] += shortage_now

        waste_now = inventory_age[i][L - 1]
        sum_waste[i] += waste_now
        inventory_age[i][L - 1] = 0

        for age_idx in range(L - 1, 0, -1):
            inventory_age[i][age_idx] = inventory_age[i][age_idx - 1]
        inventory_age[i][0] = 0

        end_inventory = sum(inventory_age[i])
        sum_end_inventory[i] += end_inventory

        prod_now = production_plan[(t, i)]
        sum_production[i] += prod_now
        weekly_capacity_used += prod_now

        if t + 1 <= max(perioden):
            pipeline[(t + 1, i)] = prod_now

        detail_rows.append({
            "Woche": t,
            "Produkt": i,
            "Echte_Nachfrage": real_demand,
            "Forecast": round(forecast[(t, i)], 4),
            "Produktion": prod_now,
            "Erfuellte_Nachfrage": fulfilled_now,
            "Fehlmenge": shortage_now,
            "Verfall": waste_now,
            "Endbestand": end_inventory
        })

    weekly_capacity_used_list.append(weekly_capacity_used / kapazitaet)

# =========================================================
# 8) KPIs
# =========================================================

kpi_rows = []

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

for i in produkte:
    kpi_rows.append({
        "Ebene": i,
        "Service_Level": sum_fulfilled[i] / sum_demand[i] if sum_demand[i] > 0 else 0,
        "Stockout_Rate": sum_shortage[i] / sum_demand[i] if sum_demand[i] > 0 else 0,
        "Durchschnittlicher_Bestand": sum_end_inventory[i] / len(perioden),
        "Verfall": sum_waste[i],
        "Produktionsauslastung": sum_production[i] / (kapazitaet * len(perioden))
    })

forecast_error_rows = []
for i in produkte:
    abs_errors = [abs(demand[(t, i)] - forecast[(t, i)]) for t in perioden]
    sq_errors = [(demand[(t, i)] - forecast[(t, i)]) ** 2 for t in perioden]

    forecast_error_rows.append({
        "Produkt": i,
        "MAE": sum(abs_errors) / len(abs_errors),
        "MSE": sum(sq_errors) / len(sq_errors)
    })

df_results = pd.DataFrame(detail_rows)
df_kpis = pd.DataFrame(kpi_rows)
df_forecast_error = pd.DataFrame(forecast_error_rows)

print("\n==== KPIs ANSATZ 4 ====")
print(df_kpis)

print("\n==== FORECAST-FEHLER ====")
print(df_forecast_error)

output_datei = str(Path(__file__).resolve().parent.parent / "results" / "ergebnisse_ansatz4_104.xlsx")

Path(output_datei).parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(output_datei, engine="openpyxl") as writer:
    df_results.to_excel(writer, sheet_name="Detailergebnisse", index=False)
    df_kpis.to_excel(writer, sheet_name="KPIs", index=False)
    df_forecast_error.to_excel(writer, sheet_name="Forecastfehler", index=False)

print(f"\nErgebnisse gespeichert unter:\n{output_datei}")