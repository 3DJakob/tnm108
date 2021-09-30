import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.array([2, 6, 5, 9], float)


print(type(arr.tolist()))

obj = pd.Series([3,5,-2,1])
print(obj)

data = pd.read_csv("ad.data",header=None, low_memory=False)
# print(data.describe())
# print(data[1:3])
print(data[data[5]>0].head(4))
data = data.replace({'?': np.nan})
data = data.replace({' ?': np.nan})
data = data.replace({'  ?': np.nan})
data = data.replace({'    ?': np.nan})
data = data.replace({'     ?': np.nan})
data = data.replace({'      ?': np.nan})
data = data.replace({'   ?': np.nan})
data = data.fillna(-1)

adindices = data[data .columns[-1]] == 'ad.'
data.loc[adindices, data .columns[-1]]=1
nonadindices = data[data .columns[-1]]=='nonad.'
data.loc[nonadindices, data .columns[-1]]=0

data[data.columns[-1]]=data[data.columns[-1]].astype(float)

data = data.apply(lambda x: pd.to_numeric(x))


# Concat

# data1 = pd.DataFrame(columns=[i for i in xrange(1559)])
# data1.loc[len(data1)] = [random.randint(0,1) for r in xrange(1558)] + [1]
# data1.loc[len(data1)] = [random.randint(0,1) for r in xrange(1558)] + [1]


# Plots
plt.plot([10,5,2,4],color='green',label='line 1',linewidth=5)
plt.xlabel('x',fontsize=40)
plt.ylabel('y',fontsize=40)
plt.axis([0,3,0,15])
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlabel('x',fontsize=40)
ax.set_ylabel('y',fontsize=40)
fig.subtitle('figure',fontsize=40)
ax.plot([10,5,2,4],color='green',label='line 1',linewidth=5)
fig.savefig('figure.png')