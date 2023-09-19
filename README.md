# ML-modeling
# This code is based on the comprehensive framework outlined in the research study titled 'Identifying Compound Binding Affinity to Corticosteroid Receptors and Selecting Important Molecular Features using SHAP Models.' It includes the complete workflow for data cleaning, dataset balancing, feature engineering, machine learning model construction, model performance comparison, and model result interpretation. This document specifically showcases the section pertaining to hyperparameter tuning in machine learning. If you are interested in the content of the code, you are welcome to contact 1120180176@mail.nankai.edu.cn.
# 创建模型字典
models = {
    'LogReg': {
        'model': LogisticRegression(random_state=2023),
        'param_grid': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'RF': {
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
    'KNN': {
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
    'MLP': {
        'model': MLPClassifier(random_state=2023),
        'param_grid': {
            'hidden_layer_sizes': [(25, 25), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    }
}

results = {}
