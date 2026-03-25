import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")

# QNN - Quantum Neural Network

"""
QNN v2 - poboljšanja:
- koristi se X_scaled u treningu
- Y za svaku poziciju se stvarno trenira u skali, 
a tek posle predikcije ide inverse_transform
- train_test_split je bez shuffle (vremenski sled).
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
svih 4586 izvlacenja
30.07.1985.- 24.03.2026.
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
df = pd.read_csv("/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv", header=None)

# ==========================
# 2. Koristimo N kombinacija
# ==========================
N = 1000
# N=len(df) # sve kombinacije
print()
print("Broj kombinacija:")
print(len(df))
print()
"""
Broj kombinacija:
4586
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
0  8  10  11  14  15  25  35
1  1   8  10  11  24  28  34
2  5   6  10  23  25  33  35
3  5   8  11  13  16  22  34
4  2  11  17  20  33  38  39
"""

print(f"Poslednjih 5 redova (od poslednjih {N}): ")
print(df_last1000.tail())
print()
"""
Poslednjih 5 redova (od poslednjih 1000): 
      0   1   2   3   4   5   6
995   1   5  11  14  15  25  39
996   7  22  23  30  31  34  38
997   1   8  11  12  29  36  39
998  17  20  27  30  31  36  37
999   1  11  25  27  31  32  39
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
# (Vremenski sled: bez shuffle-a)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.25, random_state=SEED, shuffle=False
)

print()
print(f"Primer X_train[0]: {X_train[0]}")
print(f"Primer Y_train[0]: {Y_train[0]}")
print()
"""
Primer X_train[0]: [0.31818182 0.33333333 0.2962963  0.3        0.24137931 0.51851852
 0.83333333]
Primer Y_train[0]: [ 1  8 10 11 24 28 34]
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
    y_train_pos_scaled = scaler_y_pos.fit_transform(Y_train[:, i].reshape(-1, 1))
    y_test_pos_scaled = scaler_y_pos.transform(Y_test[:, i].reshape(-1, 1))

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        # optimizer=optimizer,
        optimizer=SPSA(maxiter=300),
        loss='squared_error'
    )

    # Treniramo na sklaliranoj meti (da inverse_transform bude korektan)
    regressor.fit(X_train, y_train_pos_scaled.ravel())

    models.append(regressor)

    r2_train = regressor.score(X_train, y_train_pos_scaled.ravel())
    r2_test = regressor.score(X_test, y_test_pos_scaled.ravel())

    r2_scores_train.append(r2_train)
    r2_scores_test.append(r2_test)

    # Predikcija sledećeg broja
    pred_scaled = regressor.predict(last_scaled)
    pred = scaler_y_pos.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]
    pred = max(1, min(39, int(round(pred))))  # ograničenje 1–39

    predicted_combination.append(pred)
    print(f"Predikcija za broj {i + 1}: {pred}")
    """
    feature_map = ZFeatureMap(feature_dimension=N, reps=1)

    ansatz = TwoLocal(num_qubits=N, rotation_blocks='ry', entanglement='cz', reps=2)


    === Trening finalnog modela za POZICIJU 1 ===
    Predikcija za broj 1: 1
    === Trening finalnog modela za POZICIJU 2 ===
    Predikcija za broj 2: 2
    === Trening finalnog modela za POZICIJU 3 ===
    Predikcija za broj 3: x
    === Trening finalnog modela za POZICIJU 4 ===
    Predikcija za broj 4: y
    === Trening finalnog modela za POZICIJU 5 ===
    Predikcija za broj 5: 10
    === Trening finalnog modela za POZICIJU 6 ===
    Predikcija za broj 6: 12
    === Trening finalnog modela za POZICIJU 7 ===
    Predikcija za broj 7: z
    """


# Evaluacija
print("R² train po broju:", np.round(r2_scores_train, 4))
print("R² test  po broju:", np.round(r2_scores_test, 4))
"""
R² train po broju: [ -1.0844  -2.3786  -4.409   -6.1503  -7.2478 -12.8491 -24.0156]
R² test  po broju: [ -1.179   -2.4588  -3.8286  -5.3865  -6.0507 -10.715  -22.2238]
"""


print()
print("\n=== Predviđena sledeća loto kombinacija 7/39 ===")
print(" ".join(str(num) for num in predicted_combination))
print()

"""
N =1000

=== Predviđena sledeća loto kombinacija 7/39 ===
1 2 x y 10 12 z
"""



"""
Ključna poboljšanja :

koristi se X_scaled u treningu 
(pre je bilo računato, ali nije korišćeno),
Y za svaku poziciju se stvarno trenira u skali, 
a tek posle predikcije ide inverse_transform 
(pre je bilo skeniranje ali fit/score na neskaliranom Y, 
pa je inverse posle bio pogrešan),
train_test_split je bez shuffle (vremenski sled).
"""
