import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

# ----------------------
# 1. 加载数据集并查看基础信息
# ----------------------
# 加载数据（使用用户提供的路径）
df = pd.read_csv(r'C:\Users\Lenovo\.ssh\xiaoxueqi-666\US-pumpkins.csv')

# 查看数据基本信息
print('===== 数据基本信息 =====')
df.info()

# 查看数据集形状和前5行
rows, cols = df.shape
print(f'\n数据集形状：{rows}行 × {cols}列')
print('\n数据前5行内容：')
print(df.head())


# ----------------------
# 2. 目标变量处理（连续值转分类）
# ----------------------
# 提取目标变量（Low Price）并移除缺失值
target_col = 'Low Price'
y_continuous = df[target_col].dropna().reset_index(drop=True)  # 重置索引，确保与特征对齐

# 使用分箱将连续价格转为3个类别（低/中/高）
# 按分位数划分（确保每个类别样本量相对均衡）
discretizer = KBinsDiscretizer(
    n_bins=3,
    encode='ordinal',  # 编码为0,1,2
    strategy='quantile'
)
y = discretizer.fit_transform(y_continuous.values.reshape(-1, 1)).flatten()  # 转换为一维数组

# 输出分类结果：类别分布和价格区间
print('\n===== 目标变量分类结果 =====')
print('类别分布（0=低, 1=中, 2=高）：')
print(pd.Series(y).value_counts().sort_index())

print('\n每个类别的价格区间（左闭右开）：')
price_bins = discretizer.bin_edges_[0]  # 获取分箱边界
for i in range(3):
    print(f'类别{i}：[{price_bins[i]:.2f}, {price_bins[i+1]:.2f})')


# ----------------------
# 3. 特征处理（筛选+缺失值填充+选择）
# ----------------------
# 筛选特征：选择与价格相关的数值型特征（排除目标变量自身避免数据泄露）
feature_cols = ['High Price', 'Mostly Low', 'Mostly High']
X = df[feature_cols].loc[y_continuous.index].reset_index(drop=True)  # 与目标变量对齐

# 查看特征缺失值情况
print('\n===== 特征缺失值情况 =====')
print(X.isnull().sum())

# 填充缺失值（均值填充）
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # 转换为numpy数组

# 特征选择：选择与分类最相关的2个特征
selector = SelectKBest(score_func=f_classif, k=2)  # 分类任务用f_classif评分
X_selected = selector.fit_transform(X_imputed, y)

# 查看选中的特征名称
selected_feature_indices = selector.get_support()  # 布尔索引
selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_feature_indices[i]]
print(f'\n选中的特征：{selected_feature_names}')


# ----------------------
# 4. 划分训练集和测试集
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,  # 测试集占20%
    random_state=42,  # 固定随机种子，保证结果可复现
    stratify=y  # 按类别比例划分，避免类别不平衡
)

print('\n===== 数据集划分结果 =====')
print(f'训练集样本数：{X_train.shape[0]}（{X_train.shape[0]/len(X_selected):.0%}）')
print(f'测试集样本数：{X_test.shape[0]}（{X_test.shape[0]/len(X_selected):.0%}）')
print(f'训练集类别分布：{pd.Series(y_train).value_counts().sort_index().tolist()}')
print(f'测试集类别分布：{pd.Series(y_test).value_counts().sort_index().tolist()}')


# ----------------------
# 5. 训练随机森林分类模型
# ----------------------
# 定义模型（设置超参数，限制树深防止过拟合）
rf_model = RandomForestClassifier(
    n_estimators=100,  # 100棵决策树
    max_depth=5,       # 最大树深
    min_samples_split=10,  # 分裂节点的最小样本数
    random_state=42
)

# 训练模型
print('\n===== 模型训练开始 =====')
rf_model.fit(X_train, y_train)
print('模型训练完成！')


# ----------------------
# 6. 模型评估（分类指标）
# ----------------------
# 在测试集上预测
y_pred = rf_model.predict(X_test)

# 计算核心评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # 多分类用加权平均
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('\n===== 模型评估指标 =====')
print(f'准确率（Accuracy）：{accuracy:.4f} → 正确分类的样本占比')
print(f'精确率（Precision）：{precision:.4f} → 预测为某类且实际为该类的比例')
print(f'召回率（Recall）：{recall:.4f} → 实际为某类且被正确预测的比例')
print(f'F1分数：{f1:.4f} → 精确率和召回率的调和平均')

# 详细分类报告（每个类别的指标）
print('\n===== 详细分类报告 =====')
print(classification_report(
    y_test,
    y_pred,
    target_names=['低价', '中价', '高价']  # 自定义类别名称
))


# ----------------------
# 7. 特征重要性可视化
# ----------------------
plt.figure(figsize=(8, 5), dpi=300)
plt.bar(selected_feature_names, rf_model.feature_importances_)
plt.title('特征重要性（随机森林分类模型）', fontsize=12)
plt.ylabel('重要性得分', fontsize=10)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.tight_layout()  # 调整布局
plt.show()