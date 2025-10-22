# Data Visualization using Matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset
data = [23, 45, 56, 12, 39, 67, 34, 50]
categories = ['A', 'B', 'C', 'D', 'E']

# 1️⃣ Histogram
plt.hist(data, bins=5, edgecolor='black')
plt.title("Histogram of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Box Plot
plt.boxplot(data)
plt.title("Box Plot of Data")
plt.ylabel("Values")
plt.show()

# 3️⃣ Bar Chart
values = [23, 45, 56, 12, 39]
plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.title("Bar Chart Example")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()

# 4️⃣ Pie Chart
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title("Pie Chart Example")
plt.show()
