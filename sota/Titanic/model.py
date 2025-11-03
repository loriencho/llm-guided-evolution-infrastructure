from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
      """
      Random Forest Classifier - work by building many decision trees and combining their results to make a final prediction.
      """
      self.classifier = RandomForestClassifier() 
    
    def fit(self, X, y):
      """
      Train the model
      Args:
          X: input
          y: label
      """
      self.classifier.fit(X, y)

    def predict(self, X):
      """
      Predict the label for unseen data

      Args:
          X: unseen data

      Returns:
          The predictions for unseen data
      """
      return self.classifier.predict(X)
