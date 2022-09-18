#!/usr/bin/env python
# coding: utf-8

# In[26]:


# First let's import the packages we will use in this project
# You can do this all now or as you need them
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

#Read in the data
df = pd.read_csv(r'D:\Urvi\sql project\MovieIndustry\movies.csv')


# In[27]:


df.head()


# In[28]:


#Missing Data
for col in df.columns:
    perc_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,perc_missing))


# In[29]:


# Data Types for our columns

print(df.dtypes)


# In[30]:


#Datatye
#df["budget"] = df["budget"].astype("int64")
df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0).astype(int)
df['gross'] = pd.to_numeric(df['gross'], errors='coerce').fillna(0).astype(int)


# In[31]:


#fetch year fromm release column


# In[34]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[35]:


df.drop_duplicates()


# In[38]:


# Budget Correlation
#Company high correlation?
# Scatter Plot budges vs Gross 
plt.scatter(x=df['budget'],y=df['gross'])

plt.title('Budget v/s Gross Earning')

plt.xlabel("Gross earnings")
plt.ylabel("Film Budget")
plt.show()


# In[40]:


# Budget Correlation using Seaborn
#Company high correlation?
# Scatter Plot budges vs Gross 
sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color":"Red"},line_kws = {"color":"blue"})


# In[41]:


correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[42]:


# Using factorize - this assigns a random numeric value for each unique categorical value
df_numerized = df.apply(lambda x: x.factorize()[0]).corr()
df_numerized


# In[ ]:





# In[22]:


# df_numerized = df

# for col_name in df_numerized.columns:
#     if(df_numerized[col_name].dtype == 'object'):
#         df_numerized[col_name] = df_numerized[col_name].astype('category')
#         df_numerized[col_name] = df_numerized[col_name].cat.codes

# df_numerized


# In[43]:


df


# In[44]:


correlation_matrix = df_numerized.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[45]:


correlation_matrix1 = df_numerized.corr()
corr_pairs = correlation_matrix1.unstack()
sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[46]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[47]:


# Looking at the top 15 compaies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:


df.groupby(['company', 'year'])[["gross"]].sum()

