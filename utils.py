import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)
import wittgenstein as lw

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier


def clean_data(df):
    df['gender'] = df['gender'].replace({'Male':0, 'Female':1})
    return df

def split_data(df):
    X = df.drop(columns=['class'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, plot_roc=False, plot_confusion_matrix=False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)
    
    metrics = {
        'accuracy': np.round(accuracy_score(y_test, y_pred), 3),
        'precision': np.round(precision_score(y_test, y_pred, average='weighted'), 3),
        'recall': np.round(recall_score(y_test, y_pred, average='weighted'), 3),
        'f1_score': np.round(f1_score(y_test, y_pred, average='weighted'),3),
        'roc_auc': np.round(roc_auc_score(y_test, y_proba) if y_proba.any() else 'N/A', 3)
    }

    if plot_roc and y_proba.any():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % metrics['roc_auc'])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    if plot_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=model.classes_)
        disp.plot()
        plt.show()
    
    return metrics

def execute_knn(X_train, y_train, X_test, y_test, n_neighbors=5, plot_roc=False, plot_confusion_matrix=False):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    return model, metrics

def execute_neural_network(X_train, y_train, X_test, y_test, hidden_layer_sizes=(12,), max_iter=200, learning_rate_init=0.01, plot_roc=False, plot_confusion_matrix=False):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42, learning_rate_init=learning_rate_init)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    return model, metrics

def execute_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', plot_roc=False, plot_confusion_matrix=False):
    model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    return model, metrics

def execute_decision_tree(X_train, y_train, X_test, y_test, max_depth=None, plot_roc=False, plot_confusion_matrix=False):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    return model, metrics

def execute_ripper(X_train, y_train, X_test, y_test, plot_roc=False, plot_confusion_matrix=False):
    model = lw.RIPPER() 
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    return model, metrics
    
def plot_decision_tree(model, feature_names):
    if isinstance(model, DecisionTreeClassifier):
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=True)
        plt.show()

def apply_classifiers(X_train, y_train, X_test, y_test, plot_roc=False, plot_confusion_matrix=False):
    results = []
    
    knn_model, knn_metrics = execute_knn(X_train, y_train, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    knn_metrics['Model'] = 'KNeighborsClassifier'
    results.append(knn_metrics)
    
    svm_model, svm_metrics = execute_svm(X_train, y_train, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    svm_metrics['Model'] = 'SVC'
    results.append(svm_metrics)
    
    nn_model, nn_metrics = execute_neural_network(X_train, y_train, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    nn_metrics['Model'] = 'MLPClassifier'
    results.append(nn_metrics)
    
    tree_model, tree_metrics = execute_decision_tree(X_train, y_train, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    tree_metrics['Model'] = 'DecisionTreeClassifier'
    results.append(tree_metrics)
 
    ripper_model, ripper_metrics = execute_ripper(X_train, y_train, X_test, y_test, plot_roc=plot_roc, plot_confusion_matrix=plot_confusion_matrix)
    ripper_metrics['Model'] = 'Ripper'
    results.append(ripper_metrics)
    print(ripper_model.ruleset_)
    
    results_df = pd.DataFrame(results)
    return results_df


def df_to_tex(df):
    print(df.set_index('Model').reset_index().to_latex(float_format="{:.2f}".format))


def apply_classifiers_prob(X_train, y_train, X_test, y_test, plot_roc=False, plot_confusion_matrix=False):
    results = []

    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    lr_metrics['Model'] = 'LogisticRegression'
    results.append(lr_metrics)

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    lda_metrics = evaluate_model(lda_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    lda_metrics['Model'] = 'LinearDiscriminantAnalysis'
    results.append(lda_metrics)

    
    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)
    qda_metrics = evaluate_model(qda_model, X_test, y_test, False, False)
    qda_metrics['Model'] = 'QuadraticDiscriminantAnalysis'
    results.append(qda_metrics)

    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)
    nb_metrics = evaluate_model(nb_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    nb_metrics['Model'] = 'BernoulliNB'
    results.append(nb_metrics)

    base_learners = [
        ('lr', LogisticRegression(max_iter=200, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]
    stack_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
    stack_model.fit(X_train, y_train)
    stack_metrics = evaluate_model(stack_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    stack_metrics['Model'] = 'StackingClassifier'
    results.append(stack_metrics)

    bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
    bag_model.fit(X_train, y_train)
    bag_metrics = evaluate_model(bag_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    bag_metrics['Model'] = 'BaggingClassifier'
    results.append(bag_metrics)

    boost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), random_state=42)
    boost_model.fit(X_train, y_train)
    boost_metrics = evaluate_model(boost_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    boost_metrics['Model'] = 'AdaBoostClassifier'
    results.append(boost_metrics)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, plot_roc, plot_confusion_matrix)
    rf_metrics['Model'] = 'RandomForestClassifier'
    results.append(rf_metrics)


    results_df = pd.DataFrame(results)
    return results_df