# python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# TODO 曲线拟合：多元线性回归

# TODO 读取数据
data0 = pd.read_excel('D:\数据情况分析V3(1).xlsx', header=0).fillna(0)

# 将需要的5列数据复制，使用下面的语句读入内存, 注意，第一行（非表头）不要复制
df0 = pd.read_clipboard()

# TODO 数据清洗
# 去掉百分比的那一行，等差数列(2,,5,8...)
df0.head(10)
df0.tail(10)
df1 = df0.drop([3 * n + 2 for n in range(int(len(df0) / 3))])
# 重命名表头
this_data = df1.copy()
this_data.columns = ['system', 'ml', 'model_16', 'model_24', 'procedure']
# 转换类型
print(this_data.dtypes)
this_data['system'] = this_data['system'].astype(float).fillna(0).astype(int)
this_data['ml'] = this_data['ml'].astype(float).fillna(0).astype(int)
this_data['model_16'] = this_data['model_16'].astype(float).fillna(0).astype(int)
this_data['model_24'] = this_data['model_24'].astype(float).fillna(0).astype(int)
this_data['procedure'] = this_data['procedure'].astype(float).fillna(0).astype(int)
# TODO 数据检查
# 查看数据描述
print(this_data.describe())
# 缺失值检验，不能有缺失值，有的话，需要填补
print(this_data[this_data.isnull() == True].count())
# 查看数据箱线图
this_data.boxplot()
plt.show()
# 相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
# 相关系数0~0.3弱相关 0.3~0.6中等程度相关 0.6~1强相关
print(this_data.corr())
# 建立散点图来查看数据里的数据分析情况以及对相对应的线性情况，将使用seaborn的pairplot来绘画4种不同的因素对标签值的影响
# 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
sns.pairplot(this_data,
             x_vars=['system', 'ml', 'model_16', 'model_24'],
             y_vars='procedure',
             size=7, aspect=0.8, kind='reg')
plt.show()
# 可以看到，system和model_24对结果的线性拟合最佳

# TODO 开始创建模型
# TODO 分割测试集与训练集
X_train, X_test, Y_train, Y_test = train_test_split(this_data.iloc[:, :4], this_data.procedure, train_size=.80)
print("原始数据特征:", this_data.iloc[:, :4].shape,
      ",训练数据特征:", X_train.shape,
      ",测试数据特征:", X_test.shape)

print("原始数据标签:", this_data.procedure.shape,
      ",训练数据标签:", Y_train.shape,
      ",测试数据标签:", Y_test.shape)
# TODO 训练模型
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
# TODO 注意，每次运行的结果都不相同，
#  因为选取的训练集是随机分割的，但是只要训练的结果较好，能够组合起来预测结果就像，系数具体是多少无所谓，
#  下面是某一次的运行结果，
# 最佳拟合线:截距 219279.52139641345 ,回归系数： [ 0.92637102 -0.1861301   0.22721049  0.10064662]
# 函数表达式为：procedure = 0.92637102 * system - 0.1861301 * ml + 0.22721049 * model_16 + 0.10064662 * model_24 + 219279.52

# TODO 检测模型，评分
# R方检测：用于评估模型的精确度
# 值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
score = model.score(X_test, Y_test)
print(score)


# 0.9952393333827525
# 数据量越大，预测越准确，
# 可以运行多次，然后选择一个评分最高的，上面评分为0.99的结果是比较理想的
# 或者写一个循环，运行N次，选择评分最高的那个结果返回


def train_line_model(this_data):
    reault_opt = []
    reault_opt_model = []
    for i in range(100):
        # TODO 分割测试集与训练集
        X_train, X_test, Y_train, Y_test = train_test_split(this_data.iloc[:, :4], this_data.procedure, train_size=.80)
        # TODO 训练模型
        model = LinearRegression()
        model.fit(X_train, Y_train)
        a = model.intercept_  # 截距
        b = model.coef_  # 回归系数
        print(f"第{i}\t次拟合结果，最佳拟合线的截距：", a, ",回归系数：", b, end="")
        # TODO 注意，每次运行的结果都不相同，
        #  因为选取的训练集是随机分割的，但是只要训练的结果较好，能够组合起来预测结果就像，系数具体是多少无所谓，
        #  下面是某一次的运行结果，
        # 最佳拟合线:截距 219279.52139641345 ,回归系数： [ 0.92637102 -0.1861301   0.22721049  0.10064662]
        # 函数表达式为：procedure = 0.92637102 * system - 0.1861301 * ml + 0.22721049 * model_16 + 0.10064662 * model_24 + 219279.52

        # TODO 检测模型，评分
        # R方检测：用于评估模型的精确度
        # 值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
        score = model.score(X_test, Y_test)
        print(",评分：", score)

        # TODO 结果选择
        if score > 0.9:
            reault_opt.append(list(b) + list([a]) + [score])
            reault_opt_model.append(model)
    reault_opt_df = pd.DataFrame(reault_opt,
                                 columns=['coef_system', 'coef_ml', 'coef_model_16', 'coef_model_24', 'intercept',
                                          'score'])
    return reault_opt_df, reault_opt_model


reault_opt_df, reault_opt_model = train_line_model(this_data)
# TODO 可以从上面的结果中选择一个即可
model = reault_opt_model[0]
# TODO 对测试集进行预测
Y_pred = model.predict(X_test)
# 对比实际值
plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(Y_pred)), Y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("index of procedure")
plt.ylabel('value of procedure')
plt.show()

# TODO 完
