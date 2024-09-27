import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")

df = pd.read_csv('laptops.csv')
print(f"DF length: {len(df)}")

print(f"Number of brands: {df.nunique()['Brand']}")

mv = df.isnull().sum()
print(f"Missing values columns: \n{mv[mv>0]}\n")
print(f"Missing values: {len(mv[mv>0])}")


df['Final_Price'] = df['Final Price']
fp = df.groupby('Brand').Final_Price.max()
print(f"Dell final price: {fp['Dell']}")


median = df['Screen'].median()
print(f"Initial median: {median}")
mf_value = df['Screen'].mode()[0]
print(f"Most frequent value: {median}")
df['Screen'] = df['Screen'].fillna(mf_value)
new_median = df['Screen'].median()
print(f"New median: {new_median}")

small_df = df[df['Brand']=='Innjoo'][['RAM', 'Storage', 'Screen']]
# print(f"Innjoo df: {small_df}")
X = small_df.values
# print(f"X: \n{X}\n")

XTX = np.matmul(X.T, X)
# print(f"XTX: \n{XTX}\n")

XTX_inv = np.linalg.inv(XTX)
# print(f"XTX_inv: \n{XTX_inv}\n")

y = np.array([1100, 1300, 800, 900, 1000, 1100])
w = np.dot(np.matmul(XTX_inv, X.T), y)
print(f"W elements sum: {sum(w)}")

