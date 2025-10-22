import numpy as np
from scipy.stats import chi2_contingency

data = np.array([[207, 282], [241, 234], [242, 232]])
stat, p, dof, expected = chi2_contingency(data)
print("Chi-Square Stat:", stat)
print("p-Value:", p)
if p <= 0.05:
    print("Dependent: Reject H0")
else:
    print("Independent: H0 holds true")
