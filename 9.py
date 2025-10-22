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
