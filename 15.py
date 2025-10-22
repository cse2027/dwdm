# Compute Dissimilarity Matrix using Euclidean Distance
import pandas as pd
import numpy as np
from math import sqrt

# Create your own dataset (4 instances, 2 attributes)
data = {
    'X1': [2, 4, 5, 3],
    'X2': [10, 15, 14, 12]
}
df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")

# Compute dissimilarity matrix
n = len(df)
dissimilarity = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist = sqrt((df.loc[i, 'X1'] - df.loc[j, 'X1'])**2 +
                    (df.loc[i, 'X2'] - df.loc[j, 'X2'])**2)
        dissimilarity[i][j] = round(dist, 2)

# Convert to DataFrame
dissimilarity_df = pd.DataFrame(dissimilarity, columns=['Obj1', 'Obj2', 'Obj3', 'Obj4'], index=['Obj1', 'Obj2', 'Obj3', 'Obj4'])

print("Dissimilarity Matrix (Euclidean Distance):\n")
print(dissimilarity_df)
