import math
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
import random
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
HORIZON = 104

PRODUCTS = {
    "A": {
        "beschreibung": "stabiler Bestseller",
        "nachfrage_qual": "hoch",
        "volatilitaet_qual": "niedrig",
        "haltbarkeit_qual": "hoch",
        "batch_qual": "mittel",
        "mu": 140,
        "sigma": 1,
        "beta": 0.30,
        "alpha": 0,
        "haltbarkeit": 6,
        "batchgroesse": 40,
        "anfangsbestand": 120,
        "trend_text": "leicht steigend",
        "saison_text": "schwach",
    },
    "B": {
        "beschreibung": "volatiles Produkt mit großen Batches",
        "nachfrage_qual": "mittel",
        "volatilitaet_qual": "hoch",
        "haltbarkeit_qual": "mittel",
        "batch_qual": "hoch",
        "mu": 80,
        "sigma": 6,
        "beta": 0.00,
        "alpha": 15,
        "haltbarkeit": 4,
        "batchgroesse": 60,
        "anfangsbestand": 80,
        "trend_text": "kein Trend",
        "saison_text": "stark",
    },
    "C": {
        "beschreibung": "Nischenprodukt mit kurzer Haltbarkeit",
        "nachfrage_qual": "niedrig",
        "volatilitaet_qual": "mittel",
        "haltbarkeit_qual": "niedrig",
        "batch_qual": "niedrig",
        "mu": 20,
        "sigma": 1,
        "beta": -0.1,
        "alpha": 2,
        "haltbarkeit": 2,
        "batchgroesse": 20,
        "anfangsbestand": 40,
        "trend_text": "leicht fallend",
        "saison_text": "mittel",
    },
    "D": {
        "beschreibung": "volatiles, relativ verderbliches Produkt",
        "nachfrage_qual": "mittel",
        "volatilitaet_qual": "hoch",
        "haltbarkeit_qual": "niedrig",
        "batch_qual": "mittel",
        "mu": 60,
        "sigma": 7,
        "beta": 0.00,
        "alpha": 4,
        "haltbarkeit": 3,
        "batchgroesse": 30,
        "anfangsbestand": 60,
        "trend_text": "kein Trend",
        "saison_text": "mittel",
    },
}

def generate_demands():
    rng = random.Random(SEED)
    demands = {}
    for week in range(1, HORIZON + 1):
        for p, vals in PRODUCTS.items():
            mu = vals["mu"]
            sigma = vals["sigma"]
            beta = vals["beta"]
            alpha = vals["alpha"]
            trend = beta * week
            season = alpha * math.sin((2 * math.pi * week) / 52)
            noise = rng.uniform(-sigma, sigma)
            demand = max(0, round(mu + trend + season + noise))
            demands[(week, p)] = int(demand)
    return demands

def style_sheet(ws):
    dark = PatternFill("solid", fgColor="1F4E78")
    light = PatternFill("solid", fgColor="D9EAF7")
    border = Border(bottom=Side(style="thin", color="B7C9D6"))
    for cell in ws[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = dark
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.border = border
    for col in range(1, ws.max_column + 1):
        width = max(len(str(ws.cell(row=r, column=col).value or "")) for r in range(1, ws.max_row + 1)) + 2
        ws.column_dimensions[get_column_letter(col)].width = min(max(width, 12), 34)

def build_workbook(output_path):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Produktprofile"
    ws1.append([
        "Produkt", "Beschreibung", "Nachfrage (qual.)", "Volatilität (qual.)",
        "Haltbarkeit (qual.)", "Batchgröße (qual.)", "μ", "σ", "β", "α",
        "Haltbarkeit (Wochen)", "Batchgröße", "Anfangsbestand", "Trend", "Saison"
    ])
    for p, vals in PRODUCTS.items():
        ws1.append([
            p, vals["beschreibung"], vals["nachfrage_qual"], vals["volatilitaet_qual"],
            vals["haltbarkeit_qual"], vals["batch_qual"], vals["mu"], vals["sigma"],
            vals["beta"], vals["alpha"], vals["haltbarkeit"], vals["batchgroesse"],
            vals["anfangsbestand"], vals["trend_text"], vals["saison_text"]
        ])
    style_sheet(ws1)

    ws2 = wb.create_sheet("Generatorparameter")
    ws2.append(["Parameter", "Wert"])
    rows = [
        ("Seed", SEED),
        ("Horizont (Wochen)", HORIZON),
        ("Formel", "D(i,t) = μ(i) + β(i)*t + α(i)*sin(2πt/52) + ε(i,t)"),
        ("Zufallskomponente", "ε(i,t) ~ Uniform(-σ(i), σ(i))"),
        ("Rundung", "auf ganze Einheiten gerundet"),
        ("Untergrenze", "negative Werte werden auf 0 gesetzt"),
        ("Analysejahr", "Wochen 53-104"),
        ("Historie / Training", "Wochen 1-52"),
    ]
    for r in rows:
        ws2.append(list(r))
    style_sheet(ws2)

    ws3 = wb.create_sheet("Nachfrage_104_Wochen")
    ws3.append(["Woche", "Phase", "Produkt A", "Produkt B", "Produkt C", "Produkt D"])
    demands = generate_demands()
    for week in range(1, HORIZON + 1):
        phase = "Historie" if week <= 52 else "Analysejahr"
        ws3.append([
            week, phase,
            demands[(week, "A")],
            demands[(week, "B")],
            demands[(week, "C")],
            demands[(week, "D")],
        ])
    style_sheet(ws3)

    ws4 = wb.create_sheet("Methodik_Hinweis")
    notes = [
        "Nachfragegenerator",
        "",
        "Die Nachfrage wird additiv modelliert als:",
        "D(i,t) = μ(i) + β(i)*t + α(i)*sin(2πt/52) + ε(i,t)",
        "",
        "Bedeutung der Terme:",
        "μ(i): Basisniveau / durchschnittliche Nachfrage",
        "β(i)*t: linearer Trend",
        "α(i)*sin(2πt/52): saisonale Komponente mit 52-Wochen-Zyklus",
        "ε(i,t): gleichverteiltes Zufallsrauschen im Intervall [-σ(i), σ(i)]",
        "",
        "Implementationslogik:",
        "- deterministische Erzeugung via Seed = 42",
        "- jede Nachfrage wird auf ganze Einheiten gerundet",
        "- negative Werte werden auf 0 begrenzt",
        "- Wochen 1-52 dienen als Historie",
        "- Wochen 53-104 dienen als Analysejahr",
    ]
    for i, txt in enumerate(notes, start=1):
        ws4.cell(i, 1).value = txt
    ws4.column_dimensions["A"].width = 95
    ws4["A1"].font = Font(bold=True, size=14)
    ws4["A1"].fill = PatternFill("solid", fgColor="D9EAF7")
    ws4["A1"].alignment = Alignment(horizontal="left")

    wb.save(output_path)

def plot_demand(output_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_excel(output_path, sheet_name="Nachfrage_104_Wochen")

    pivot_df = df[["Woche", "Produkt A", "Produkt B", "Produkt C", "Produkt D"]].copy()

    plt.figure(figsize=(14, 7))

    colors = {
        "Produkt A": "blue",
        "Produkt B": "green",
        "Produkt C": "red",
        "Produkt D": "purple"
    }

    for col in ["Produkt A", "Produkt B", "Produkt C", "Produkt D"]:
        plt.plot(
            pivot_df["Woche"],
            pivot_df[col],
            label=col,
            color=colors[col],
            linewidth=2
        )

    plt.title("Generierte Nachfrage über 104 Wochen", fontsize=14)
    plt.xlabel("Woche", fontsize=12)
    plt.ylabel("Nachfrage", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plot_path = output_path.with_name("plot_1_nachfrageverlauf.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot gespeichert: {plot_path}")

if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent.parent / "data" / "demand_104_setup2.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_workbook(output_path)
    print(f"Excel-Datei erstellt: {output_path}")

plot_demand(output_path)