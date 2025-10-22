import pandas as pd
import numpy as np

# Create simulated dataset with unique instances
np.random.seed(10)
data = {
    'ID': np.arange(1, 11),
    'Age': np.random.randint(20, 50, size=10),
    'Salary': np.random.randint(30000, 90000, size=10),
    'City': np.random.choice(['Hyderabad', 'Chennai', 'Bangalore', 'Pune'], size=10)
}

df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
print("Simulated Unique Dataset:\n", df)

"""
OUTPUT:

Simulated Unique Dataset:
    ID  Age  Salary       City
0   1   29   47904  Hyderabad
1   2   49   72909       Pune
2   3   24   46241  Hyderabad
3   4   35   73002  Hyderabad
4   5   20   39224       Pune
5   6   37   39289       Pune
6   7   47   85552  Bangalore
7   8   48   61210  Hyderabad
8   9   45   58712       Pune
9  10   49   40742  Bangalore
"""
