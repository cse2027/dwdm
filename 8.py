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
