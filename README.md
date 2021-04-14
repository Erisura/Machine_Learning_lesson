# Machine_Learning_lesson
the experiments of Machine Learning lesson
实验使用`python 3.6`和`sklearn`完成
## Experiment 1: Linear Regression

###实验任务：
  使用波士顿房价预测模型进行线性回归
  1. 数据读入<br/>方式1：`data = pd.read_csv('boston.csv')`<br/>方式2：`data = sklearn.datasets.load_boston()`
  2. 定义特征值和目标值<br/>根据题目要求，定义如下特征值和目标值(其他值的相关性太低，不予考虑)：<br/>`data_used = data[['crim', 'rm', 'lstat', 'medv']]`
  3. 特征值的统计性描述<br/>(25%、50%、75%、max为四分位点)
![image](https://user-images.githubusercontent.com/72057715/114651721-1100b600-9d17-11eb-9d58-4db2152c7dff.png)
  5. 进行训练和预测<br/>
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("回归方程系数:{}".format(lr.coef_))
print("回归方程截距:{}".format(lr.intercept_))

y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)
```
  5.用MAE和MSE两种评价指标进行评价:<br/>
```
test_mae = mean_absolute_error(y_pred_test, y_test)
test_mse = mean_squared_error(y_pred_test, y_test)
print("test mae: " + str(test_mae))
print("test mse: " + str(test_mse))

train_mae = mean_absolute_error(y_pred_train, y_train)
train_mse = mean_squared_error(y_pred_train, y_train)
print("train_mae: " + str(train_mae))
print("train_mse: " + str(train_mse))
```
  6.结果可视化:<br/>
```
plt.scatter(y_test, y_pred_test)
plt.plot([0, 50], [0, 50])  # (0,0)到(50,50)的线 K为黑色
plt.show()
```

## Experiment2: Support Vector Machine

### 使用iris数据集进行支持向量机预测实验
<br/>由于整体实验步骤和上述实验类似，不再详细介绍
<br/>实验中，分别使用了`rbf`、`linear`、`poly`三种内核
详细可见代码

## Experiment3: Clustering

### 使用iris数据集进行聚类实验
<br/>分别使用了`K-means`、`高斯混合模型(使用EM算法完成迭代)`、`谱聚类`完成实验
<br/>分别的聚类效果如下:
![原始类别](https://user-images.githubusercontent.com/72057715/114652558-b23c3c00-9d18-11eb-9687-8ae9bf7fe098.png)
![K-means](https://user-images.githubusercontent.com/72057715/114652569-b6685980-9d18-11eb-83a6-7b40ab0058e4.png)
![GMM](https://user-images.githubusercontent.com/72057715/114652572-ba947700-9d18-11eb-8b2d-0e1849d7ed1e.png)
![SC](https://user-images.githubusercontent.com/72057715/114652584-be27fe00-9d18-11eb-9ce9-01e4745d42af.png)

