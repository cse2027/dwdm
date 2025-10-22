import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread'],
    ['milk', 'bread', 'apple'],
    ['nuts', 'apple', 'milk']
]

te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

frequent_items = apriori(df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

print("Frequent Itemsets:\n", frequent_items)
print("\nAssociation Rules:\n", rules)

"""
OUTPUT:

Frequent Itemsets:
     support              itemsets
0       0.8               (apple)
1       0.8               (bread)
2       1.0                (milk)
3       0.4                (nuts)
4       0.6        (bread, apple)
5       0.8         (milk, apple)
6       0.4         (apple, nuts)
7       0.8         (milk, bread)
8       0.4          (milk, nuts)
9       0.6  (milk, bread, apple)
10      0.4   (milk, apple, nuts)

Association Rules:
        antecedents    consequents  antecedent support  ...  jaccard  certainty  kulczynski
0          (bread)        (apple)                 0.8  ...      0.6      -0.25        0.75
1          (apple)        (bread)                 0.8  ...      0.6      -0.25        0.75
2           (milk)        (apple)                 1.0  ...      0.8       0.00        0.90
3          (apple)         (milk)                 0.8  ...      0.8       0.00        0.90
4           (nuts)        (apple)                 0.4  ...      0.5       1.00        0.75
5           (milk)        (bread)                 1.0  ...      0.8       0.00        0.90
6          (bread)         (milk)                 0.8  ...      0.8       0.00        0.90
7           (nuts)         (milk)                 0.4  ...      0.4       0.00        0.70
8    (milk, bread)        (apple)                 0.8  ...      0.6      -0.25        0.75
9    (milk, apple)        (bread)                 0.8  ...      0.6      -0.25        0.75
10  (bread, apple)         (milk)                 0.6  ...      0.6       0.00        0.80
11         (bread)  (milk, apple)                 0.8  ...      0.6      -0.25        0.75
12         (apple)  (milk, bread)                 0.8  ...      0.6      -0.25        0.75
13    (milk, nuts)        (apple)                 0.4  ...      0.5       1.00        0.75
14   (apple, nuts)         (milk)                 0.4  ...      0.4       0.00        0.70
15          (nuts)  (milk, apple)                 0.4  ...      0.5       1.00        0.75

[16 rows x 14 columns]
"""
