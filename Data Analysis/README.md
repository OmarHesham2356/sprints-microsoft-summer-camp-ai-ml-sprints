# Quick reference — NumPy & Pandas: code + short comments (for documenting your sprints)

Nice — I turned your course topics into compact code snippets with comments so you can paste them straight into your repo `README` or sprint notebooks. Each snippet is bite-sized and focused on *what to remember*.

---

## NumPy — setup & intro

```python
# install (run in terminal)
# pip install numpy

# import
import numpy as np  # standard alias
```

---

## NumPy basics — arrays, dtypes, shape

```python
a = np.array([1, 2, 3])          # 1D array
b = np.array([[1,2,3],[4,5,6]])  # 2D array (matrix)
print(a.dtype, a.shape)          # dtype, shape
c = np.arange(0, 10, 2)          # [0,2,4,6,8]
d = np.linspace(0, 1, 5)         # 5 values evenly spaced between 0 and 1
zeros = np.zeros((3,2))          # zeros matrix 3x2
ones = np.ones(4)                # 1D ones
eye = np.eye(3)                  # identity matrix 3x3
```

---

## Creating arrays (from lists / files / other)

```python
# from Python lists
lst = [10,20,30]
arr = np.array(lst)

# from file (CSV, whitespace)
# numeric data only: use numpy.loadtxt
# np.loadtxt('data.csv', delimiter=',', skiprows=1)

# from pandas (if mixed types or headers)
# import pandas as pd
# df = pd.read_csv('data.csv')
# arr = df[['col1','col2']].to_numpy()
```

---

## Arrays manipulation (reshape, transpose, flatten, copy)

```python
x = np.arange(12)          # 0..11
X = x.reshape(3,4)         # reshape to 3x4 (must keep same total elements)
Xt = X.T                   # transpose
flat = X.ravel()           # view flattened
copy_flat = X.flatten()    # copy flattened
X2 = X.reshape(-1,2)       # -1 infers dimension automatically
```

---

## Stacking & splitting arrays

```python
a = np.array([1,2,3])
b = np.array([4,5,6])
vstacked = np.vstack([a,b])    # stack rows -> 2x3
hstacked = np.hstack([a,b])    # join horizontally -> 1x6
col = a.reshape(-1,1)
conc = np.concatenate([col, col], axis=1)

# splitting
parts = np.split(np.arange(8), [3,6])  # splits at indices -> [0:3], [3:6], [6:]
```

---

## Math operations & ufuncs

```python
x = np.array([1., 2., 3.])
y = np.array([4., 5., 6.])
z = x + y               # element-wise add
prod = x * y            # multiply element-wise
dot = x.dot(y)          # dot product (1D)
mat = X @ X.T           # matrix multiplication
np.sum(X, axis=0)       # sum per column
np.mean(X, axis=1)      # mean per row
np.sqrt(x)              # universal function (ufunc)
np.exp(x)               # exponential
```

---

## Boolean indexing & fancy indexing

```python
arr = np.array([10, 5, 8, 20])
mask = arr > 8               # boolean mask
arr[mask]                    # selects values > 8
# fancy indexing with integer lists
idx = [3, 0, 2]
arr[idx]                     # reorder/select by indices
```

---

## Types & casting

```python
a = np.array([1,2,3], dtype=np.int32)
b = a.astype(np.float64)   # cast to float
# safe casting: np.can_cast(src_dtype, dst_dtype)
```

---

## Random numbers

```python
rng = np.random.default_rng(42)  # new Generator API (recommended)
sample = rng.random(5)           # 5 floats in [0,1)
ints = rng.integers(0, 10, size=5)  # random ints 0..9
choice = rng.choice([10,20,30], size=3, replace=True)
```

---

## Linear algebra basics (with numpy.linalg)

```python
from numpy.linalg import inv, det, eig, solve

A = np.array([[3,1],[2,4]])
b = np.array([7,10])

x = solve(A, b)      # solve A x = b
A_inv = inv(A)       # inverse matrix
values, vectors = eig(A)  # eigenvalues & eigenvectors
detA = det(A)        # determinant
```

---

## Advanced NumPy — reshaping, advanced indexing, broadcasting

```python
# broadcasting example (vector + matrix)
M = np.ones((3,4))
v = np.array([1,2,3,4])
M_plus_v = M + v      # v broadcasts across rows

# advanced indexing: combining boolean & integer
rows = np.array([0,2])
cols = np.array([1,3])
M[np.ix_(rows, cols)]  # select submatrix with outer indexing
```

---

## Pandas — setup & basics

```python
# pip install pandas
import pandas as pd

# Series (1D labeled)
s = pd.Series([10, 20, 30], index=['a','b','c'])

# DataFrame (2D)
df = pd.DataFrame({
    'name': ['ali','mona','tariq'],
    'age': [23, 25, 22],
    'score': [88, 92, 79]
})
```

---

## Creating Series & DataFrames (from dicts, lists, numpy)

```python
# from dict of lists (most common)
data = {'col1':[1,2], 'col2':[3,4]}
df = pd.DataFrame(data)

# from numpy
arr = np.arange(6).reshape(3,2)
df2 = pd.DataFrame(arr, columns=['x','y'])
```

---

## Importing data (CSV / Excel / JSON)

```python
df = pd.read_csv('data.csv')             # CSV (most common)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')  # Excel
df = pd.read_json('data.json')           # JSON
# tips: use parse_dates=['date_col'] to auto-parse dates
```

---

## Data overview & quick checks

```python
df.head()           # first 5 rows
df.tail(3)          # last 3 rows
df.info()           # dtypes, non-null counts
df.describe()       # summary stats for numeric columns
df['col'].value_counts()  # frequency counts
```

---

## Indexing & selection (loc, iloc, at, iat)

```python
df.loc[0, 'name']        # label-based selection
df.loc[0:2, ['name','age']]  # rows 0..2
df.iloc[0, 2]            # integer position-based
df.at[0, 'name']         # fast scalar by label
df.iat[0, 2]             # fast scalar by position
```

---

## Handling missing data

```python
df.isna().sum()          # count missing per column
df.dropna(subset=['col'], inplace=False)  # drop rows missing in 'col'
df.fillna({'age': df['age'].median()})    # fill missing with median
df.interpolate()         # linear interpolation for numeric columns
```

---

## Shape-shifting: melt, pivot, stack/unstack

```python
# wide -> long
long = pd.melt(df, id_vars=['name'], value_vars=['score','age'],
               var_name='metric', value_name='value')

# pivot (long -> wide)
pivot = long.pivot(index='name', columns='metric', values='value')

# stack/unstack for hierarchical indexes
stacked = df.set_index(['name','age']).stack()
```

---

## Grouping & aggregation

```python
# group by a column and aggregate
grouped = df.groupby('age')['score'].mean()   # mean score per age

# multi-aggregation
agg = df.groupby('age').agg({'score':['mean','max'], 'name':'count'})

# transform (returns same shape as original)
df['score_z'] = df.groupby('age')['score'].transform(lambda x: (x - x.mean())/x.std())
```

---

## Advanced aggregation & pivot\_table

```python
pivot = pd.pivot_table(df, index='age', columns='name', values='score',
                       aggfunc='mean', fill_value=0)
```

---

## DataFrame operations — merge, join, concat, apply

```python
# concat (stack dataframes)
big = pd.concat([df1, df2], axis=0, ignore_index=True)

# merge (SQL-style join)
merged = df_left.merge(df_right, on='id', how='left')

# apply row-wise or column-wise functions
df['len_name'] = df['name'].apply(len)      # element-wise
df['score_scaled'] = df['score'].apply(lambda x: x/100)
```

---

## Time series basics

```python
df['date'] = pd.to_datetime(df['date_str'])   # convert
df = df.set_index('date')
df.resample('M').mean()                       # monthly mean
df['rolling_7d'] = df['value'].rolling(window=7).mean()
```

---

## Vectorization & NumPy + Pandas together

```python
# use numpy functions on pandas Series (fast)
df['sqrt_score'] = np.sqrt(df['score'].to_numpy())

# boolean mask with numpy
mask = np.logical_and(df['age'] > 20, df['score'] > 80)
df[mask]
```

---

## Feature engineering intro (simple examples)

```python
# binning numeric -> categorical
df['age_group'] = pd.cut(df['age'], bins=[0,18,25,100], labels=['kid','young','adult'])

# one-hot encoding
dummies = pd.get_dummies(df['category'], prefix='cat')

# date features
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
```

---

## Comparing Pandas with Polars (short note to remember)

* Pandas = feature-rich, widely used, great for many tasks.
* Polars = faster, multi-threaded, lower memory for big datasets (useful when you hit scale limits).
  (Write a short personal note: “try Polars if pandas becomes too slow with big files.”)

---

