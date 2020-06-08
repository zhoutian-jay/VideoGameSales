
# coding: utf-8

# # Video Games Sales
# 
# 
# 数据文件：vgsales.csv
# 
# 该数据集包含游戏名称、类型、发行时间、发布者以及在全球各地的销售额数据。
# 
# 数据量：11列共1.66W数据。
# 
# 基于这个数据集，可进行以下问题的探索：
# 
# 电子游戏市场分析：受欢迎的游戏、类型、发布平台、发行人等；
# 预测每年电子游戏销售额。
# 可视化应用：如何完整清晰地展示这个销售故事。
# 也可以自行发现其他问题，并进行相应的挖掘。

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# # 加载数据并显示数据摘要

# Treat Platform, Genre, and Publisher as categorical data

# In[2]:


dtypes = {'Platform':'category',
          'Genre':'category',
          'Publisher':'category'}


# Load the data and ignore games that are missing data

# In[10]:


df = (pd.read_csv('vgsales.csv',dtype=dtypes).dropna())
df.head()


# 将年份转化为整型数据

# In[8]:


df['Year'] = (df['Year']
              .astype('int64'))


# In[9]:


df.head()


# In[6]:


df.info(memory_usage='deep')


# In[8]:


df['Genre'].unique()


# In[9]:


df['Publisher'].unique()


# In[10]:


df['Platform'].unique()


# # 几个游戏市场的差异分析

# 'NA_Sales'市场最大

# In[11]:


df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].agg('sum')


# 日本游戏市场的概况

# In[12]:


df[df['JP_Sales'] > 1]['Publisher'].value_counts()[:10]


# # 不同风格的游戏在各大市场的趋势分析

# In[13]:


market_sales = (df
                .groupby('Genre')
                .agg('sum')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']])


# In[14]:


for column in market_sales.columns:
    market_sales[column + '_frac'] = market_sales[column] / df[column].agg('sum')


# 不同风格游戏在各大市场表现的可视化

# In[15]:


(market_sales[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
 .plot
 .bar(rot=25));


# 把每个类型的销售额标准化到市场的总销售额。角色扮演类游戏在日本最受欢迎，而动作类、体育类和射击类游戏在其他市场中排名前三

# In[16]:


(market_sales[['NA_Sales_frac', 'EU_Sales_frac', 'JP_Sales_frac', 'Other_Sales_frac']]
 .plot
 .bar(rot=25));


# # 一款电子游戏的最能说明它的成功特性分析

# 相关性分析

# In[19]:


df_ohe = pd.get_dummies(df, columns=dtypes.keys())


# In[20]:


corrs = df_ohe.corr()


# In[64]:


corrs['Global_Sales'].sort_values(ascending=False)[:15]


# In[63]:


corrs['Global_Sales'].sort_values()[:10]


# 使用 Random Forest 来判断特征的重要性

# In[34]:


from sklearn.preprocessing import OneHotEncoder


# In[35]:


enc = OneHotEncoder()


# In[36]:


X_raw = df[['Year', 'Publisher', 'Genre', 'Platform']]
X = enc.fit_transform(X_raw)

y = df['Global_Sales'].values


# In[37]:


from sklearn.ensemble import RandomForestRegressor


# In[38]:


from sklearn.model_selection import GridSearchCV, KFold


# In[46]:


rfr = RandomForestRegressor(n_estimators=100)
params = {'max_depth':[3, 5, 10]}


# In[47]:


rf_gs = GridSearchCV(rfr,
                     param_grid=params,
                     scoring='neg_mean_squared_error',
                     cv=KFold(n_splits=5, shuffle=True),
                     return_train_score=False)


# In[48]:


rf_gs.fit(X, y)


# In[49]:


rf_gs.best_estimator_


# In[50]:


feat_impt = pd.Series(dict(zip(enc.get_feature_names(),
                               rf_gs.best_estimator_.feature_importances_)))


# In[66]:


feat_impt.sort_values(ascending=False)[:15]


# In[52]:


year_sales = (df
              .groupby('Year')
              .agg(['mean', 'std'])['Global_Sales'])


# In[53]:


df = df.merge(year_sales,
              how='outer',
              left_on='Year',
              right_index=True)


# In[54]:


df['Global_Sales_zScore'] = (df['Global_Sales'] - df['mean']) / df['std']


# In[57]:


X_raw = df[df['Year'] < 2019][['Publisher', 'Genre', 'Platform']]
X = enc.fit_transform(X_raw)

y = df[df['Year'] < 2019]['Global_Sales_zScore'].values


# In[59]:


rfr = RandomForestRegressor(n_estimators=100,
                            max_depth=5)
rfr.fit(X, y)


# In[61]:


feat_impt_Z = pd.Series(dict(zip(enc.get_feature_names(),
                                 rfr.feature_importances_)))


# In[67]:


feat_impt_Z.sort_values(ascending=False)[:15]

