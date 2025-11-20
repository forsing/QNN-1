import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")



"""
QNN
"""


"""
| Paket                       | Verzija |
| --------------------------- | ------- |
| **python**                  | 3.11.13 |
| **qiskit**                  | 1.4.4   |
| **qiskit-machine-learning** | 0.8.3   |
| **qiskit-ibm-runtime**      | 0.43.0  |
| **macOS**                   | Tahos   |
| **Apple**                   | M1      |
"""


"""
https://github.com/forsing
https://github.com/forsing?tab=repositories
"""


"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4506 izvlacenja
30.07.1985.- 04.11.2025.
"""



from qiskit_aer.primitives import EstimatorV2

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.optimizers import ADAM
from qiskit.circuit.library import TwoLocal, ZFeatureMap
from qiskit import QuantumCircuit, transpile

from sklearn.model_selection import train_test_split

from sklearn import model_selection
import numpy as np
import pandas as pd
import random

from qiskit_machine_learning.utils import algorithm_globals

from sklearn.preprocessing import MinMaxScaler

from qiskit_algorithms.gradients import ParamShiftEstimatorGradient


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


# Učitavanje svih kombinacija iz CSV fajla
df = pd.read_csv("/data/loto7_4506_k87.csv", header=None)

# ==========================
# 2. Koristimo N kombinacija
# ==========================
N = 1000 
# N=len(df) # sve kombinacije

print()
print("Broj kombinacija:")
print(N)
print()
"""
Broj kombinacija:
1000
"""


# Uzima poslednjih N kombinacija
df_last1000 = df.tail(N).reset_index(drop=True)


print()
print(f"Prvih 5 redova (od poslednjih {N}): ")
print(df_last1000.head())
print()
"""
Prvih 5 redova (od poslednjih 1000):
   0   1   2   3   4   5   6
0  8  11  15  22  33  38  39
1  4  10  15  20  26  31  37
2  2   6  11  22  28  31  34
3  4   6  13  21  24  32  38
4  5  15  18  28  30  33  36
"""


print(f"Poslednjih 5 redova (od poslednjih {N}): ")
print(df_last1000.tail())
print()
"""
Poslednjih 5 redova (od poslednjih 1000):
     0   1   2   3   4   5   6
995  1   3  11  12  19  35  38
996  1   2   6  10  18  24  34
997  6  20  21  22  27  29  36
998  5  19  23  30  31  34  36
999  6   9  12  20  24  31  36
"""


data = df_last1000.values

# X - sve osim poslednje
X = data[:-1]

# Y - sledeće pomereno za 1
Y = data[1:]


# Skaliranje X
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

# Poslednja kombinacija za predikciju
last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)


# Podela na trening i test skup
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=SEED
)

print()
print(f"Primer X_train[0]: {X_train[0]}")
print(f"Primer Y_train[0]: {Y_train[0]}")
print()
"""
Primer X_train[0]: [13 23 24 29 33 35 39]
Primer Y_train[0]: [ 3 15 19 24 26 34 39]
"""


# ============ QNN setup ============

N = 7  # broj qubita

feature_map = ZFeatureMap(feature_dimension=N, reps=1)
ansatz = TwoLocal(num_qubits=N, rotation_blocks='ry', entanglement='cz', reps=2)

# kombinuje feature_map i ansatz u jedan QC
qc = QuantumCircuit(N)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

# 🔹 ključno: transpile u osnovni skup gejtova koje Aer razume
qc = transpile(qc, basis_gates=["rx", "ry", "rz", "cx", "id"])

# ✅ lokalni simulator
# shots (default je 1024)
estimator_v2 = EstimatorV2()

gradient = ParamShiftEstimatorGradient(estimator=estimator_v2)

# sada kreiraj QNN
qnn = EstimatorQNN(
    estimator=estimator_v2,
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    gradient=gradient
)

optimizer = COBYLA(maxiter=300, tol=1e-6)
# optimizer = ADAM(maxiter=300, lr=0.1)

# ============ treniranje po svakom broju ============

models = []
r2_scores_train = []
r2_scores_test = []

predicted_combination = []

for i in range(7):
    print(f"=== Trening finalnog modela za POZICIJU {i + 1} ===")

    # Skaliranje Y za tu poziciju
    scaler_y_pos = MinMaxScaler(feature_range=(0, 1))
    y_scaled_pos = scaler_y_pos.fit_transform(Y[:, i].reshape(-1, 1))

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        # optimizer=optimizer,
        optimizer = SPSA(maxiter=300),
        loss='squared_error'
    )

    regressor.fit(X_train, Y_train[:, i])

    models.append(regressor)
    
    r2_train = regressor.score(X_train, Y_train[:, i])
    r2_test = regressor.score(X_test, Y_test[:, i])
    
    r2_scores_train.append(r2_train)
    r2_scores_test.append(r2_test)


    # Predikcija sledećeg broja
    pred_scaled = regressor.predict(last_scaled)
    pred = scaler_y_pos.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    pred = max(1, min(39, int(round(pred))))  # ograničenje 1–39

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i + 1}: {pred}")
    """
    === Trening finalnog modela za POZICIJU 1 ===
    Predikcija za broj 1: 1
    === Trening finalnog modela za POZICIJU 2 ===
    Predikcija za broj 2: 2
    ...
    """


# Evaluacija
print("R² train po broju:", np.round(r2_scores_train, 4))
print("R² test  po broju:", np.round(r2_scores_test, 4))
"""
R² train po broju: [  -2.3891   -1.7549   -5.1671  -10.6939  -39.7657  -65.5059 -776.1818]
R² test  po broju: [ -1.6974  -3.9422  -6.4919 -10.7985 -20.4284 -37.9215 -98.3278]
"""


print()
print("\n=== Predviđena sledeća loto kombinacija 7/39 ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""
1000
=== Predviđena sledeća loto kombinacija 7/39 ===
1 2 x x x 11 15
"""
