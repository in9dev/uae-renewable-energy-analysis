import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

df = pd.read_csv("Data/uae_renewable_energy.csv")


def exploration(df):
    """Print first few rows and dataset dimensions to explore the data"""
    print("\nEXPLORATION PROCESS:\n")

    print("First 3 rows:")
    print(df.head(3))
    
    print(f"\n(row, column): {df.shape}\n")

    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected!\n", missing)

def analyze(df):
    """Uses Pandas to extract and calculate data"""
    df["Solar_Increase"] = df["Solar_Capacity_MW"].diff()
    max_increase_year = df.loc[df["Solar_Increase"].idxmax(), "Year"]
    print("Year with largest increase in solar capacity:", max_increase_year)

    growth = df["Clean_Energy_Share_%"].iloc[-1] - df["Clean_Energy_Share_%"].iloc[0]
    print("Growth in clean energy share:", growth, "%")

    correlation = df["Clean_Energy_Share_%"].corr(df["CO2_Reduction_Mt"])
    print("Correlation:", correlation)

def plotAnalysis(df):
    """Plot analysis using matplotlib"""
    plt.plot(df["Year"], df["Solar_Capacity_MW"], label="Solar", marker="o")
    plt.plot(df["Year"], df["Wind_Capacity_MW"], label="Wind", marker="s")
    plt.xlabel("Year")
    plt.ylabel("Capacity (MW)")
    plt.title("UAE Renewable Energy Growth")
    plt.legend()
    plt.savefig("plotAnalysis.png")
    plt.show()


def plotPredictionSolar(df):
    """Calculate solar predictions to get an estimate for following years, then plotting."""
    def exp_func(x, a, b, c):
        return a * np.exp(b * (x - 2015)) + c

    X = df["Year"].values
    y = df["Solar_Capacity_MW"].values

    params, _ = curve_fit(exp_func, X, y)
    future = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031])
    preds = exp_func(future, *params)

    print(preds)
    
    # Plot actual data
    plt.plot(df["Year"], df["Solar_Capacity_MW"], marker='o', color='blue', label="Actual Solar Capacity")

    # Plot exponential fit for all years including future
    all_years = np.append(X, future)
    all_preds = exp_func(all_years, *params)
    plt.plot(all_years, all_preds, marker='x', linestyle='--', color='red', label="Exponential Fit & Prediction")

    # Annotate predicted points
    for year, pred in zip(future, preds):
        plt.text(year, pred+50, f"{int(pred)} MW", color='red')

    # Labels and title
    plt.xlabel("Year")
    plt.ylabel("Solar Capacity (MW)")
    plt.title("UAE Solar Capacity: Actual vs Exponential Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("plotSolarPrediction.png")
    plt.show()


def plotPredictionWind(df):
    """Calculate wind predictions to get an estimate for following years, then plotting."""
    def exp_func(x, a, b, c):
        return a * np.exp(b * (x - 2015)) + c

    X = df["Year"].values
    y = df["Wind_Capacity_MW"].values

    params, _ = curve_fit(exp_func, X, y)
    future = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031])
    preds = exp_func(future, *params)

    print(preds)
    
    # Plot actual data
    plt.plot(df["Year"], df["Wind_Capacity_MW"], marker='o', color='blue', label="Actual Wind Capacity")

    # Plot exponential fit for all years including future
    all_years = np.append(X, future)
    all_preds = exp_func(all_years, *params)
    plt.plot(all_years, all_preds, marker='x', linestyle='--', color='red', label="Exponential Fit & Prediction")

    # Annotate predicted points
    for year, pred in zip(future, preds):
        plt.text(year, pred+50, f"{int(pred)} MW", color='red')

    # Labels and title
    plt.xlabel("Year")
    plt.ylabel("Wind Capacity (MW)")
    plt.title("UAE Wind Capacity: Actual vs Exponential Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("plotWindPrediction.png")
    plt.show()


def plotCO2Reduction(df):
    """Plot (Bar Chart) the CO2 Reduction"""
    plt.bar(df["Year"], df["CO2_Reduction_Mt"], label="CO2 Reduction")
    plt.xlabel("Year")
    plt.ylabel("CO2 Reduction (Mt)")
    plt.title("UAE CO2 Reduction Growth")
    plt.legend()
    plt.savefig("plotCO2Reduction.png")
    plt.show()


def plotPredictionCO2Reduction(df):
    """Calculate CO2 reduction predictions for future years, then plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit

    def exp_func(x, a, b, c):
        return a * np.exp(b * (x - 2015)) + c

    X = df["Year"].values
    y = df["CO2_Reduction_Mt"].values

    # Fit exponential curve
    params, _ = curve_fit(exp_func, X, y)

    # Future years to predict
    future = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031])
    preds = exp_func(future, *params)
    print("Predicted CO2 Reduction (Mt) for future years:", preds)

    # Plot actual CO2 reduction as bars
    plt.bar(X, y, color='green', label="Actual CO2 Reduction")

    # Plot exponential fit including future predictions
    all_years = np.append(X, future)
    all_preds = exp_func(all_years, *params)
    plt.plot(all_years, all_preds, marker='x', linestyle='--', color='red', label="Exponential Fit & Prediction")

    # Annotate predicted points
    for year, pred in zip(future, preds):
        offset = pred * 0.05  # 5% of the predicted value
        plt.text(year + 0.1, pred+offset, f"{pred:.1f} Mt", color='red', rotation=45)

    # Labels and title
    plt.xlabel("Year")
    plt.ylabel("CO2 Reduction (Mt)")
    plt.title("UAE CO2 Reduction: Actual vs Predicted (Exponential)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plotCO2Prediction.png")
    plt.show()




while True:
    print("""
1) Explore the data
2) Analyze the data (summary analysis)
3) Plot the data (solar/wind)
4) Plot the prediction (solar)
5) Plot the prediction (wind)
6) Plot the CO2 Reduction
7) Plot the prediction (CO2 Reduction)
9) Quit
    """)
    userInp = int(input("Choose an option: "))

    match userInp:
        case 1:
            exploration(df)
        case 2:
            analyze(df)
        case 3:
            plotAnalysis(df)
        case 4: 
            plotPredictionSolar(df)
        case 5: 
            plotPredictionWind(df)
        case 6:
            plotCO2Reduction(df)
        case 7:
            plotPredictionCO2Reduction(df)
        case 9:
            break
        case _:
            print("Invalid option, try again.")
