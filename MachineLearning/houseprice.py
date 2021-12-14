import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("train.csv", index_col=0)
test_data = pd.read_csv("test.csv", index_col=0)
#输出训练集测试集前几行
print(train_data.head(),test_data.head())
#输出训练集，测试集的大小
print(train_data.shape)
print(test_data.shape)
#输出训练集测试集信息
print(train_data.info)
print(test_data.info)
df=pd.DataFrame(train_data.dtypes,columns=['类型'])
print(df.head(10))
df=pd.DataFrame(test_data.dtypes,columns=['类型'])
print(df.head())
#房价直方图
sns.displot(train_data['SalePrice'])
plt.show()
#查看特征之间关联程度：相关系数矩阵可视化
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap='magma')
plt.show()
#查看Saleprice相关程度较强的几个变量的混淆矩阵
#取10个corr最大的变量
cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
corrSP = np.corrcoef(train_data[cols].values.T)
hm = sns.heatmap(corrSP,cmap='magma',annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()
#数据可视化
#SaleType
data=pd.concat([train_data['SalePrice'],train_data['SaleType']],axis=1)
sns.boxplot(x='SaleType',y='SalePrice',data=data)
plt.show()
#GarageType
data=pd.concat([train_data['SalePrice'],train_data['GarageType']],axis=1)
sns.boxplot(x='GarageType',y='SalePrice',data=data)
plt.show()
#YearBuilt
data=pd.concat([train_data['SalePrice'],train_data['YearBuilt']],axis=1)
plt.scatter(x='YearBuilt',y='SalePrice',data=data)
plt.show()
#LotArea
data=pd.concat([train_data['SalePrice'],train_data['LotArea']],axis=1)
plt.scatter(x='LotArea',y='SalePrice',data=data)
plt.show()
#MasVnrArea
data=pd.concat([train_data['SalePrice'],train_data['MasVnrArea']],axis=1)
sns.scatterplot(x='MasVnrArea',y='SalePrice',data=data)
plt.show()
#YearRemodAdd
data=pd.concat([train_data['SalePrice'],train_data['YearRemodAdd']],axis=1)
sns.barplot(x='YearRemodAdd',y='SalePrice',data=data)
plt.xticks([])
plt.show()
#LotFrontage
data=pd.concat([train_data['SalePrice'],train_data['LotFrontage']],axis=1)
sns.boxplot(x='LotFrontage',y='SalePrice',data=data)
plt.xticks([])
plt.show()
#异常值
fig, ax = plt.subplots()
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#删除异常值
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
#检查异常值
fi, ax = plt.subplots()
ax.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#用原始数据中的房价和取对数后的房价创建了个新的DataFrame
prices = pd.DataFrame({"price": train_data["SalePrice"], "log(price + 1)": np.log1p(train_data["SalePrice"])})
# 把train_data的特征SalePrice提取出来
y_train = np.log1p(train_data.pop("SalePrice"))
print(y_train.head())
#将train_data和test_data放在一起处理其他特征
all_data = pd.concat((train_data, test_data), axis=0)
#查看all_data的大小
print(all_data.shape)
#删除重复数据并查看删除后的前几行
all_data.drop_duplicates()
print(all_data.head())
#查看缺失值
print(train_data.isnull().sum())
#缺失率
all_data_na = (all_data.isnull().sum() / len(all_data))
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'缺失率': all_data_na})
print(missing_data)
#可视化
f, ax = plt.subplots(figsize=(10,8))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
#缺失值处理
#删除
all_data.drop(['PoolQC','Alley','MiscFeature','Fence','Utilities'],axis=1,inplace=True)
#用None填充
cols1 = [ "FireplaceQu", "GarageQual", "GarageCond",
         "GarageFinish",  "GarageType", "BsmtExposure",
         "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
         "MasVnrType","MasVnrArea","MSSubClass"]
for col in cols1:
    all_data[col].fillna("None", inplace=True)
#用0填充
cols=[ "BsmtUnfSF",  "GarageCars","GarageYrBlt","TotalBsmtSF",
      "BsmtFinSF2", "BsmtFinSF1", "GarageArea","BsmtFullBath","BsmtHalfBath"]
for col in cols:
    all_data[col].fillna(0, inplace=True)
#均值填充
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#特殊值
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#检查是否还有缺失值
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'缺失率' :all_data_na})
print(missing_data.head())

#变量转换
all_data['OverallQual'] = all_data['OverallQual'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
# 使用get_dummies方法可将all_data中特征值是字符串的特征转成one-hot形式编码
all_dummy_data = pd.get_dummies(all_data)

# 对所有数据进行归一化
numeric_cols = all_data.columns[all_data.dtypes != "object"]
numeric_col_means = all_dummy_data.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_data.loc[:, numeric_cols].std()
all_dummy_data.loc[:, numeric_cols] = (all_dummy_data.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# 数据重新分回训练集和测试集
dummy_train_data = all_dummy_data.loc[train_data.index]
dummy_test_data = all_dummy_data.loc[test_data.index]

# .values将dataframe对象转换成numpy array形式
X_train = dummy_train_data.values
X_test = dummy_test_data.values

#模型
# 设定随机森林中的决策树使用的特征占比
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
# 网格搜索来寻找最佳max_feat
# 使用随机森林模型预测
for max_feat in max_features:
#n_estimators为最大弱学习器的个数(决策树的个数),max_features=max_feat即决策树使用的特征占所有特征的比例
     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
# cross_val_score即交叉验证方法,cv=5代表5折,neg_mean_squared_error即负均方误差
     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring="neg_mean_squared_error"))
# 记录得到的test_score平均值
     test_scores.append(np.mean(test_score))
plt.subplots(figsize=(10, 8))
plt.plot(max_features, test_scores)
plt.xlabel("max_features")
plt.ylabel("test_scores")
plt.show()
# 根据上面网格搜索,使用随机森林时最佳max_features=0.3
rf = RandomForestRegressor(n_estimators=200, max_features=.3)
rf.fit(X_train, y_train)
y_rf = np.expm1(rf.predict(X_test))
submission_data = pd.DataFrame(data={"Id": test_data.index, "SalePrice": y_rf})
submission_data.to_csv("submission.csv", index=False)
print(submission_data.head(20))
