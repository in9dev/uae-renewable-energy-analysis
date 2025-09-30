# UAE Renewable Energy Analysis

An interactive Python project that explores, analyzes, visualizes, and predicts the growth of solar and wind energy in the UAE, along with CO₂ reduction trends.

---

## Project Overview

This project demonstrates **data exploration, analysis, visualization, and prediction** using Python.  
It includes:

- Interactive menu for exploration and visualization.
- Data analysis and summary statistics.
- Visualizations of solar and wind energy growth.
- CO₂ reduction analysis.
- Exponential curve fitting to predict future values (2024-2031).

The project uses **Pandas**, **Matplotlib**, **NumPy**, and **SciPy** for data handling and analysis.

---

## Dataset

The dataset contains UAE renewable energy statistics:

| Column | Description |
|--------|-------------|
| `Year` | Year of record |
| `Solar_Capacity_MW` | Solar power capacity in megawatts |
| `Wind_Capacity_MW` | Wind power capacity in megawatts |
| `Clean_Energy_Share_%` | Share of clean energy in total production |
| `CO2_Reduction_Mt` | Reduction in CO₂ emissions in megatons |

> The CSV file is stored in the `data/` folder.

---

## Options
```
1) Explore the data
2) Analyze the data (summary analysis)
3) Plot the data (solar/wind)
4) Plot the prediction (solar)
5) Plot the prediction (wind)
6) Plot the CO2 Reduction
7) Plot the prediction (CO2 Reduction)
9) Quit
```
