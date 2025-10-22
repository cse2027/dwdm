# Apriori Algorithm using Python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactional data
dataset = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread'],
    ['milk', 'bread', 'apple'],
    ['milk', 'bread', 'nuts']
]

# Convert to DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
