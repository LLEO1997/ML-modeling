#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE


# X_pca，top_mipi 和 X_wed。这些数据可以作为机器学习算法的输入

# In[2]:


# 读取数据
df = pd.read_excel('D:/LYC_ML/数据爬取/ANN_ML/五分类/CE605240.xlsx')
names = df['name']  # Assuming 'name' is the column name in your df


# In[3]:


df.info()
df.sample(3)


# In[4]:


# 将“GR”列中的非数字行替换为空值并去除
df.dropna(subset=['GR (LN)'], inplace=True)

# 数据清洗和特征选择
data=df.iloc[:,4:]
# 将字符串替换为空值
data = data.applymap(lambda x: np.nan if isinstance(x, str) else x)
# 删除空值超过90%的列
threshold = 0.9 * len(data)
data.dropna(axis=1, thresh=threshold, inplace=True)
# 数据清洗和去重
data.drop_duplicates(inplace=True)
data = data.loc[:, data.apply(pd.Series.nunique) != 1]  # 移除方差为零的特征
corr_matrix = data.corr().abs()  # 计算特征间的皮尔逊相关系数
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]  # 去除具有高成对相关性的特征
data.drop(to_drop, axis=1, inplace=True)


# In[5]:


data.info()
data.sample(3)


# In[6]:


# 删除包含NaN的行
data.dropna(inplace=True)

# 重置索引
data.reset_index(drop=True, inplace=True)

# 对目标变量进行编码
label_encoder = LabelEncoder()
data['GR (LN)'] = label_encoder.fit_transform(data['GR (LN)'])

# 分离特征和目标变量
X = data.iloc[:, 1:]
y = data['GR (LN)']

# 特征标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 只选择类0的数据进行聚类
class_0_data = X_scaled[y == 0]
# 删除类0数据中的重复数据点
class_0_data_unique = class_0_data.drop_duplicates()
# 选择约500个中心点
n_clusters_actual = 500
kmeans = KMeans(n_clusters=n_clusters_actual, random_state=2023).fit(class_0_data_unique)

cluster_centers = kmeans.cluster_centers_

# 从原始数据中移除所有类0的数据
X_no_class_0 = X_scaled[y != 0]
y_no_class_0 = y[y != 0]

# 使用SMOTE对1、3、4类进行过采样到约500个样本
smote = SMOTE(sampling_strategy={1: 500, 3: 500, 4: 500}, random_state=2023)
X_smoted, y_smoted = smote.fit_resample(X_no_class_0, y_no_class_0)

# 将KMeans的聚类中心加入到已过采样的数据中
X_scaled = pd.concat([X_smoted, pd.DataFrame(cluster_centers, columns=X.columns)], axis=0)
y = pd.concat([y_smoted, pd.Series([0]*n_clusters_actual)], axis=0)

print(y.value_counts())


# In[7]:


# # 输出重采样表格
# # 读取已保存的df
# original_df = pd.read_excel('D:/LYC_ML/数据爬取/ANN_ML/五分类/CE605240.xlsx')
# saved_columns = original_df[['name', 'code']]
# # 根据X_scaled和y的索引获取name和code列
# final_saved_columns = saved_columns.loc[X_scaled.index]


# In[8]:


# # 重置索引
# final_saved_columns_reset = final_saved_columns.reset_index(drop=True)
# y_reset = y.reset_index(drop=True)
# X_scaled_reset = X_scaled.reset_index(drop=True)

# # 使用已重置索引的数据框执行concat操作
# final_data = pd.concat([final_saved_columns_reset, y_reset, X_scaled_reset], axis=1)

# # 将数据保存到Excel文件中
# final_data.to_excel('D:/LYC_ML/数据爬取/ANN_ML/五分类/Processed_CE605240.xlsx', index=False)


# In[9]:


# 使用SimpleImputer填充缺失值
imputer = SimpleImputer(strategy='mean')
X_scaled_imputed_np = imputer.fit_transform(X_scaled)
X_scaled = pd.DataFrame(X_scaled_imputed_np, columns=X.columns)  # 再次转换回DataFrame并保留列名


# In[11]:


from sklearn.model_selection import KFold
# 设置随机数种子
random_state = 2023

# RFE递归消除
# 假设 X_scaled 和 y 已经被定义并准备好
# 使用支持向量机作为基础模型
svc = SVC(kernel="linear", random_state=random_state)

# RFECV 执行了 RFE 并使用交叉验证选择最佳数量的特征
rfecv = RFECV(estimator=svc, step=20, cv=KFold(5, shuffle=True, random_state=random_state), scoring='accuracy')

# Fit RFECV to the data
X_rfecv = rfecv.fit_transform(X_scaled, y)

# 找到被选中的特征
selected_features_rfecv = X_scaled.columns[rfecv.support_]

# 创建新的dataframe，只包含被选中的特征
X_rfecv = pd.DataFrame(X_rfecv, columns=selected_features_rfecv)


# In[12]:


# kbest
np.random.seed(2023)  # 设置随机数种子
# 计算你希望保留的特征数量（即原始特征数量的一半）
num_features = X_scaled.shape[1]
num_features_to_select = num_features // 2  # 使用整数除法，结果为不大于原始结果的最大整数

# 使用计算出的特征数量创建 SelectKBest
kbest = SelectKBest(k=num_features_to_select)

# 训练 SelectKBest 并转换数据
X_kbest = kbest.fit_transform(X_scaled, y)

# 获取所选特征的掩码
mask = kbest.get_support()

# 从原始特征中选择被选中的特征
new_features = X_scaled.columns[mask]

# 使用所选的特征创建新的 DataFrame
X_kbest = pd.DataFrame(X_kbest, columns=new_features)


# In[13]:


# 互信息（MI）
mi_func = lambda *args, **kwargs: mutual_info_regression(*args, **kwargs, random_state=random_state)
mi = SelectPercentile(mi_func, percentile=50)
X_mi = mi.fit_transform(X_scaled, y)
# 获取MI得分
mi_scores = mi.scores_

# 选择得分大于0.01的特征
selected_idx_mi_threshold = [idx for idx, score in enumerate(mi_scores) if score > 0.01]

# 获取选定的特征名
selected_cols_mi_threshold = X_scaled.columns[selected_idx_mi_threshold]

# 筛选DataFrame以仅包括选定的特征
X_mi_threshold = X_scaled[selected_cols_mi_threshold]

# 如果需要，可以将其转换为新的DataFrame
X_mi = pd.DataFrame(X_mi_threshold, columns=selected_cols_mi_threshold)

# 排列重要性（PI）
model = RandomForestRegressor(n_jobs=8, random_state=random_state)
model.fit(X_scaled, y)

# 计算排列重要性得分
result = permutation_importance(model, X_scaled, y, n_repeats=10, n_jobs=8)
scores = result.importances_mean

# 获取PI得分（您已经有了这些得分，存在变量'scores'中）
pi_scores = scores

# 选择得分大于0的特征
selected_idx_pi_threshold = [idx for idx, score in enumerate(pi_scores) if score > 0]

# 获取选定的特征名
selected_cols_pi_threshold = X_scaled.columns[selected_idx_pi_threshold]

# 筛选DataFrame以仅包括选定的特征
X_pi_threshold = X_scaled[selected_cols_pi_threshold]

# 如果需要，可以将其转换为新的DataFrame
X_pi = pd.DataFrame(X_pi_threshold, columns=selected_cols_pi_threshold)


# In[14]:


print(X_rfecv.shape)

print(X_kbest.shape)

print(X_mi.shape)

print(X_pi.shape)

print(y.shape)


# In[15]:


# 创建模型字典
models = {
    '逻辑回归': {
        'model': LogisticRegression(random_state=2023),
        'param_grid': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    '随机森林': {
        'model': RandomForestClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 4, 6]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    'K近邻': {
        'model': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    '多层感知器': {
        'model': MLPClassifier(random_state=2023),
        'param_grid': {
            'hidden_layer_sizes': [(25, 25), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    }
}

results = {}


# In[16]:


import warnings
warnings.filterwarnings(action='ignore')


# In[17]:


# 训练和评估模型
for model_name, model_info in models.items():
    print(f'正在训练 {model_name}...')
    model = model_info['model']
    param_grid = model_info['param_grid']
    for input_type, input_data in zip(['all', 'RFE', 'kbest', 'MI', 'PI'], [X_scaled, X_rfecv, X_kbest, X_mi, X_pi]):
        X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.2, random_state=random_state)
        grid_search = GridSearchCV(model, param_grid, cv=KFold(10, shuffle=True, random_state=random_state))
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_pred = best_model.predict(input_data)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        total_accuracy = accuracy_score(y, y_pred)  # 计算整体准确率

        classification_report_train = classification_report(y_train, y_train_pred)
        classification_report_test = classification_report(y_test, y_test_pred)
        total_classification_report = classification_report(y, y_pred)  # 计算整体分类报告

        class_names_train = np.array(y_train)
        y_train_binarize = label_binarize(y_train, classes=class_names_train)
        y_train_fit=label_binarize(y_train_pred, classes = class_names_train)
        fpr_train, tpr_train, _=metrics.roc_curve(y_train_binarize.ravel(),y_train_fit.ravel())
        auc_train = metrics.auc(fpr_train, tpr_train)
        
        class_names_test = np.array(y_test)
        y_test_binarize = label_binarize(y_test, classes=class_names_test)
        y_test_fit=label_binarize(y_test_pred, classes = class_names_test)
        fpr_test, tpr_test, _=metrics.roc_curve(y_test_binarize.ravel(),y_test_fit.ravel())
        auc_test = metrics.auc(fpr_test, tpr_test)

        class_names = np.array(y)
        y_binarize = label_binarize(y, classes=class_names)
        y_fit = label_binarize(y_pred, classes = class_names)
        fpr, tpr, _ = metrics.roc_curve(y_binarize.ravel(), y_fit.ravel())
        auc_total = metrics.auc(fpr, tpr)  # 计算整体AUC

        if model_name not in results:
            results[model_name] = {}

        results[model_name][input_type] = {
            '训练准确率': train_accuracy,
            '测试准确率': test_accuracy,
            '整体准确率': total_accuracy,
            '训练auc': auc_train,
            '测试auc': auc_test,
            '整体auc': auc_total,
            '训练分类报告': classification_report_train,
            '测试分类报告': classification_report_test,
            '整体分类报告': total_classification_report,
            '最佳模型': best_model
        }


# In[18]:


# 输出结果
data_for_csv = []

for model_name, input_types in results.items():
    print(f'\n{model_name}')
    print('{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<30} {:<30} {}'.format('输入类型', '训练准确率', '测试准确率', '整体准确率', '训练auc', '测试auc', '整体auc', '训练分类报告', '测试分类报告', '整体分类报告'))
    for input_type, result in input_types.items():
        train_accuracy = '{:.4f}'.format(result['训练准确率'])
        test_accuracy = '{:.4f}'.format(result['测试准确率'])
        total_accuracy = '{:.4f}'.format(result['整体准确率'])  # 新增
        auc_train =  '{:.4f}'.format(result['训练auc'])
        auc_test =  '{:.4f}'.format(result['测试auc'])
        total_auc =  '{:.4f}'.format(result['整体auc'])  # 新增
        classification_report_train = result['训练分类报告']
        classification_report_test = result['测试分类报告']
        total_classification_report = result['整体分类报告']  # 新增
        print('{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<30} {:<30} {}'.format(input_type, train_accuracy, test_accuracy, total_accuracy, auc_train, auc_test, total_auc, classification_report_train, classification_report_test, total_classification_report))
        
        data_for_csv.append([model_name, input_type, train_accuracy, test_accuracy, total_accuracy, auc_train, auc_test, total_auc, classification_report_train, classification_report_test, total_classification_report])

# 将结果保存为CSV
df_1 = pd.DataFrame(data_for_csv, columns=['模型名', '输入类型', '训练准确率', '测试准确率', '整体准确率', '训练auc', '测试auc', '整体auc', '训练分类报告', '测试分类报告', '整体分类报告'])
df_1.to_csv('D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/模型结果.csv', index=False)


# In[19]:


name_mapping = {
    '逻辑回归': 'Logistic regression',
    '随机森林': 'RF',
    'XGBoost模型': 'XGBoost',
    'K近邻': 'KNN',
    'AdaBoost模型': 'AdaBoost',
    '多层感知器': 'MLP'
}

# 更改models字典的键
models = {name_mapping.get(key, key): value for key, value in models.items()}

# 同时更改results字典的键
results = {name_mapping.get(key, key): value for key, value in results.items()}


# In[20]:


# 混淆矩阵
import matplotlib.pyplot as plt
import os

# 定义保存图像的目录
save_directory = 'D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/混淆矩阵'

# 确保目录存在
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for model_name, model_info in models.items():
    best_model = results[model_name]['PI']['最佳模型'] 
    disp = plot_confusion_matrix(best_model, X_test, y_test)
    disp.ax_.set_title(f'Confusion Matrix for {model_name}')
    
    # 保存图像到指定目录
    plt.savefig(os.path.join(save_directory, f'{model_name}_Confusion_Matrix.png'))
    
    plt.show()
    


# In[21]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

# 将标签二值化
y_bin = label_binarize(y_test, classes=np.unique(y_test))

# 初始化一个存放所有假阳性率和真阳性率的字典
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 10))

for model_name, model_info in models.items():
    best_model = results[model_name]['PI']['最佳模型'] 
    y_score = best_model.predict_proba(X_test)
    # 对每个类别，计算ROC曲线和AUC
    for i in range(y_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # 计算微平均ROC曲线和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_bin.shape[1])]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(y_bin.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= y_bin.shape[1]
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=f'{model_name} ROC micro-average (area = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/roc_curve.png')
plt.show()


# In[22]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt



# 使用 label_binarize 将 'y' 转化为二进制形式
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = y_test_binarized.shape[1]

# 设置图像大小
plt.figure(figsize=(10, 10))

# 计算每个类别的 Precision-Recall 并绘制曲线
for i in range(n_classes):
    for model_name, model_info in models.items():
        best_model = results[model_name]['PI']['最佳模型']
        y_score = best_model.predict_proba(X_test)[:, i]
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score)
        average_precision = average_precision_score(y_test_binarized[:, i], y_score)
        plt.plot(recall, precision, label=f'{model_name} Precision-Recall curve of class {i} (area = {average_precision:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for multi-class data using micro-average')
plt.legend(loc="lower right")
plt.savefig('D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/Precision-Recall curve.png')
plt.show()


# In[23]:


# 使用 label_binarize 将 'y' 转化为二进制形式
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = y_test_binarized.shape[1]

# 计算每个类别的 Precision-Recall 并绘制曲线
for i in range(n_classes):
    # 设置图像大小
    plt.figure(figsize=(10, 10))
    
    for model_name, model_info in models.items():
        best_model = results[model_name]['PI']['最佳模型']
        y_score = best_model.predict_proba(X_test)[:, i]
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score)
        average_precision = average_precision_score(y_test_binarized[:, i], y_score)
        plt.plot(recall, precision, label=f'{model_name} Precision-Recall curve of class {i} (area = {average_precision:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for class {i}')
    plt.legend(loc="lower right")
    plt.savefig(f'D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/Precision-Recall curve_class_{i}.png')
    plt.show()


# In[24]:


import seaborn as sns
# 类别分布图
sns.countplot(x=y)
plt.title('Class Distribution')
plt.savefig('D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/Class Distribution.png')
plt.show()


# In[26]:


import shap

best_model = results['XGBoost']['PI']['最佳模型']
explainer = shap.TreeExplainer(best_model)

# 然后用这个explainer计算SHAP值
shap_values = explainer.shap_values(X_pi)


# In[27]:


import os
import matplotlib.pyplot as plt

# 定义保存图像的路径
save_path = r'D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote'

# 确保路径存在
os.makedirs(save_path, exist_ok=True)

# 对于每个类别绘制并保存SHAP图像
for i in range(5):
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values[i], X_pi, plot_type="dot", max_display=10, show=False)
    plt.savefig(os.path.join(save_path, f'shap_class_{i}.png'), dpi=300)  # 保存图像，设置分辨率为200 dpi
    plt.clf()


# In[28]:


# 定义保存CSV的路径
csv_path =  r'D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote'

# 确保路径存在
os.makedirs(csv_path, exist_ok=True)

# 对于每个类别，计算并保存特征的SHAP值
for i in range(5):
    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap_values = np.mean(np.abs(shap_values[i]), axis=0)

    # 将特征和它们的重要性对应起来
    feature_importances = dict(zip(X_pi.columns, mean_abs_shap_values))

    # 按重要性对特征进行排序
    sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    # 打印排序后的特征重要性
    for feature, importance in sorted_feature_importances:
        print(f"Feature: {feature}, Importance: {importance}")

    # 创建一个DataFrame对象
    df_importances = pd.DataFrame(sorted_feature_importances, columns=['Feature', 'Importance'])

    # 将DataFrame保存为CSV文件
    df_importances.to_csv(os.path.join(csv_path, f'feature_importances_{i}.csv'), index=False)


# In[29]:


# 用SHAP计算所有数据的平均shap值
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# 数据清洗和特征选择的函数
def data_preprocessing(df):
    # 数据清洗和特征选择
    data=df.iloc[:,4:]
    # 将字符串替换为空值
    data = data.applymap(lambda x: np.nan if isinstance(x, str) else x)
    # 删除空值超过90%的列
    threshold = 0.9 * len(data)
    data.dropna(axis=1, thresh=threshold, inplace=True)
    # 数据清洗和去重
    data.drop_duplicates(inplace=True)
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]  # 移除方差为零的特征
    corr_matrix = data.corr().abs()  # 计算特征间的皮尔逊相关系数
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]  # 去除具有高成对相关性的特征
    data.drop(to_drop, axis=1, inplace=True)
    # 删除包含NaN的行
    data.dropna(inplace=True)
    return data
# 读取数据
df = pd.read_excel('D:/LYC_ML/数据爬取/ANN_ML/五分类/CE605240.xlsx')


# In[30]:


# 将“GR”列中的非数字行替换为空值并去除
data = data_preprocessing(df)
names = df.loc[data.index, 'name'].reset_index(drop=True)

# 数据清洗和特征选择
data = data_preprocessing(df)

# 重置索引
data.reset_index(drop=True, inplace=True)

# 对目标变量进行编码
label_encoder = LabelEncoder()
data['GR (LN)'] = label_encoder.fit_transform(data['GR (LN)'])

# 分离特征和目标变量
X = data.iloc[:, 1:]
y = data['GR (LN)']

# 找出共有的列
common_columns = list(set(X.columns) & set(X_pi.columns))

# 根据 X_pi 的列顺序来对这些共有的列进行排序
common_columns.sort(key=lambda x: list(X_pi.columns).index(x))

# 选择共有的列
X = X[common_columns]
X_pi = X_pi[common_columns]

# 特征标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 建立SHAP解释器
best_model = results['XGBoost']['PI']['最佳模型']
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_scaled)

# 获取模型的基线预测值
baseline_predictions = explainer.expected_value

# 计算每个类别的SHAP得分
shap_scores = [np.sum(vals, axis=1) for vals in shap_values]

# 创建一个新的DataFrame
df_with_SHAP = pd.DataFrame({
    'Name': names,
    'Actual_GR_LN': y,
})
# N 是 df_with_SHAP 的行数
N = len(df_with_SHAP)

for i, baseline in enumerate(baseline_predictions):
    df_with_SHAP[f'Baseline_Prediction_Class_{i}'] = [baseline] * N  # 重复 baseline 以填满 N 行
for i, scores in enumerate(shap_scores):
    df_with_SHAP[f'SHAP_Score_Class_{i}'] = scores

# 添加最终预测类别
final_predictions = best_model.predict(X_scaled)
df_with_SHAP['Final_Class'] = final_predictions

# 保存到CSV文件
df_with_SHAP.to_csv('D:/LYC_ML/数据爬取/ANN_ML/五分类/20230915_kmeans+smote/df_with_SHAP.csv', index=False)


# In[ ]:




