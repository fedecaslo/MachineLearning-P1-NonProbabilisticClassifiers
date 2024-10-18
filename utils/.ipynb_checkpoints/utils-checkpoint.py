from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def apply_classifiers(X_train, y_train, X_val, y_val):
    knn = KNeighborsClassifier(n_neighbors=5) 
    knn.fit(X_train, y_train)
    y_val_pred_knn = knn.predict(X_val)
    knn_accuracy = accuracy_score(y_val, y_val_pred_knn)
    print(f'Precisión de KNN en validación: {knn_accuracy:.4f}')
    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_val_pred_svm = svm.predict(X_val)
    svm_accuracy = accuracy_score(y_val, y_val_pred_svm)
    print(f'Precisión de SVM en validación: {svm_accuracy:.4f}')
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)  
    mlp.fit(X_train, y_train)
    y_val_pred_mlp = mlp.predict(X_val)
    mlp_accuracy = accuracy_score(y_val, y_val_pred_mlp)
    print(f'Precisión de Red Neuronal en validación: {mlp_accuracy:.4f}')
    
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_val_pred_tree = tree.predict(X_val)
    tree_accuracy = accuracy_score(y_val, y_val_pred_tree)
    print(f'Precisión de Árbol de Clasificación en validación: {tree_accuracy:.4f}')
