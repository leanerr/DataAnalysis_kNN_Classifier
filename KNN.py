#!/usr/bin/env python
# coding: utf-8

# # KNN

# In[111]:


import pandas as pd
import numpy as np


# In[80]:


#فراخوانی دیتاست
data= pd.read_excel('./dataset2.xls',header=None)
data.head()


# # هدف مشخص کردن دسته مربوط به فیچر لیبل دار میباشد(آخرین فیچر) 

# In[ ]:


X=data.values[:, 0:10]
y=data.values[:, 10]


# دیتاهایمان را به دو دسته برای تست و ترین مشخص میکنیم

# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[110]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score


# # فرآیند انتخاب فیچر های مناسب

#  با این کار ممکن است بتوانیم با انتخاب فیچر های مناسب دقت پیشبینی مدل خود را افزایش دهیم 

# In[103]:


class SBS():
    
    def __init__(self, estimator, k_features, scoring=accuracy_score,               
                 test_size=0.2, random_state=1):     
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
        
    def fit(self, X, y):
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:                 
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# In[104]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
#k_features = 1 
#به این معناست که یک به یک فیچر ها را بررسی کن
sbs.fit(X_train, y_train)


# In[105]:


#نمودار زیر دقت پیشبینی برای تعداد فیچر ها را نشان میدهد
#به ما کمک میکند که متوجه شویم با استفاده از چند فیچر ، دقت پیش بینی ما بالاتر است
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()


# In[106]:


#برای استفاده از تمامی فیچر ها
k = list(sbs.subsets_[0])
print(data.columns[0:10][k])


# In[107]:


#برای استفاده از تعداد فیچر مناسب
#تعدادی از فیچر ها را حذف میکند 
k2 = list(sbs.subsets_[3])
#سه تا از فیچر ها حذف میگردند
print(data.columns[0:10][k2])


# In[108]:


# بدون در نظر گرفتن انتخاب فیچر 
knn.fit(X_train, y_train)
print('Training accuracy:', knn.score(X_train, y_train))
print('Test accuracy:', knn.score(X_test, y_test))


# In[109]:


# با درنظر گرفتن انتخاب فیچر
knn.fit(X_train[:, k2], y_train)
print('Training accuracy:', knn.score(X_train[:, k2], y_train))
print('Test accuracy:', knn.score(X_test[:, k2], y_test))


# علی عسگری-9611415026
