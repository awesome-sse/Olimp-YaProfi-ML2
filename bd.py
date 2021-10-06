import pandas as pd

df = pd.read_csv('formula_train.csv')

df1 = pd.read_csv('train.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df1.head())
print(df1.info())
print(df1.describe())

for i in range(1,82):
    sr = (df1['feature'+str(i)].max()+df1['feature'+str(i)].min())/2
    df['feature'+str(i)] = (df1['feature'+str(i)]-sr)**2/sr
df['critical_temperature'] = df1['critical_temperature']

del df['material']

print(df.info())
print(df.describe())


df.to_csv('train1.csv', sep=',')
