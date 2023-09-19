#!/usr/bin/env python
# coding: utf-8

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
