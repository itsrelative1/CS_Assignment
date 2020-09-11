import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from matplotlib.patches import Rectangle

data = pd.read_csv("istherecorrelation.csv", sep=";", decimal=",")

Year = data["Year"]
WO = data["WO [x1000]"].values
Cons = data["NL Beer consumption [x1000 hectoliter]"].values

# Scale values by mappin 0, 1 to min and max
scaler = preprocessing.MinMaxScaler()
WO_scaled = scaler.fit_transform(WO.reshape(-1, 1))
Cons_scaled = scaler.fit_transform(Cons.reshape(-1, 1))

pearson_corr_matrix = np.corrcoef(WO_scaled.ravel(), Cons_scaled.ravel())
pearson_corr_value = pearson_corr_matrix[0, 1]

fig, ax = plt.subplots(1, 1)
(l1,) = ax.plot(Year, WO_scaled)
(l2,) = ax.plot(Year, Cons_scaled)
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)

ax.legend(
    (extra, l1, l2),
    (
        "Pearson correlation = " + f"{round(pearson_corr_value, 3)}",
        "WO normalized",
        "NL Beer consumption normalized",
    ),
)

ax.set_xlabel("Year")
plt.savefig("drink", dpi=300)
plt.show()

