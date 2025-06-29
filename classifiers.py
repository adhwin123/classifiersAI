import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

class ClassifierManager:
    def __init__(self, data):
        '''
        This initializes the class with the data and preprocesses it for model training.
        '''
        self.features, self.labels = self._preprocess_data(data)
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(self.features, self.labels)
        self.results = []

    def _preprocess_data(self, data):
        '''
        Preprocess the input data, separating features and labels.
        '''
        features = data.iloc[:, :-1].values
        labels = data.iloc[:, -1].values
        return features, labels

    def _split_data(self, features, labels, test_size=0.4):
        '''
        Split the dataset into training and testing sets.
        '''
        return train_test_split(features, labels, test_size=test_size, random_state=42)

    def evaluate_classifier(self, classifier, param_grid, model_name):
        '''
        Evaluate the given classifier using grid search and evaluate its performance.
        '''
        # Perform grid search
        grid_search = self._perform_grid_search(classifier, param_grid)
        
        # Extract the best model and score
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Train and evaluate the model
        training_score = best_model.score(self.X_train, self.y_train)
        test_score = best_model.score(self.X_test, self.y_test)
        
        # Print results
        print(f"Classifier: {model_name}")
        print(f"Best Score: {best_score}")
        print(f"Training Score: {training_score}")
        print(f"Test Score: {test_score}")
        print(f"Best Parameters: {best_params}")
        
        # Plot the decision boundary
        self._plot_decision_boundary(self.X_test, self.y_test, best_model, model_name)
        
        # Store results
        self.results.append(f"{model_name}, {best_score}, {test_score}")

    def _perform_grid_search(self, classifier, param_grid):
        '''
        Perform grid search with cross-validation for the given classifier and parameter grid.
        '''
        grid_search = GridSearchCV(classifier, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search

    def _plot_decision_boundary(self, X, y, model, model_name):
        '''
        Plot the decision boundary for a given classifier.
        '''
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.02

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)
        plt.title(model_name)
        plt.savefig(f"{model_name}_boundary.png")
        plt.show()


class ClassifierFactory:
    def create_knn(self):
        '''
        Create K-Nearest Neighbors classifier with grid search parameters.
        '''
        param_grid = {
            'n_neighbors': range(1, 21, 2),
            'leaf_size': range(5, 31, 5)
        }
        return KNeighborsClassifier(), param_grid

    def create_logistic_regression(self):
        '''
        Create Logistic Regression classifier with grid search parameters.
        '''
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [100, 200, 300]
        }
        return LogisticRegression(), param_grid

    def create_decision_tree(self):
        '''
        Create Decision Tree classifier with grid search parameters.
        '''
        param_grid = {
            'max_depth': range(1, 51),
            'min_samples_split': range(2, 11)
        }
        return DecisionTreeClassifier(), param_grid

    def create_random_forest(self):
        '''
        Create Random Forest classifier with grid search parameters.
        '''
        param_grid = {
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_split': range(2, 11)
        }
        return RandomForestClassifier(), param_grid

    def create_adaboost(self):
        '''
        Create AdaBoost classifier with grid search parameters.
        '''
        param_grid = {
            'n_estimators': range(10, 71, 10)
        }
        return AdaBoostClassifier(), param_grid


def main():
    data = pd.read_csv('input.csv')
    manager = ClassifierManager(data)
    factory = ClassifierFactory()
    
    # List of classifiers to evaluate
    classifiers = [
        ("KNN", factory.create_knn()),
        ("Logistic Regression", factory.create_logistic_regression()),
        ("Decision Tree", factory.create_decision_tree()),
        ("Random Forest", factory.create_random_forest()),
        ("AdaBoost", factory.create_adaboost())
    ]
    
    # Evaluate each classifier
    for name, (clf, param_grid) in classifiers:
        manager.evaluate_classifier(clf, param_grid, name)
    
    # Save results to CSV
    with open("output.csv", "w") as f:
        f.write("Classifier, Best Training Score, Testing Score\n")
        for result in manager.results:
            f.write(result + '\n')


if __name__ == "__main__":
    main()
