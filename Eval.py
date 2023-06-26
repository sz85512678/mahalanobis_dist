class Eval:
    """
    Evaluate an anomaly detection method given corpus and test set
    """
    def __init__(self, detector=None, corpus=None, inlier=None, outlier=None):
        """

        :param detector: an object with method fit and predict
        :param corpus: Training set as a list of stream
        :param inlier: Testing set, inliers, as a list of stream
        :param outlier: Testing set, inliers, as a list of stream(np.array)
        """
        self.corpus = corpus
        self.inlier = inlier
        self.outlier = outlier
        self.detector = detector

    def eval(self):
        """
        :return: (accuracy, precision, recall, false positive rate, false negative rate)
        """
        inlier_predictions = [self.detector.predict(inlier) for inlier in self.inlier]
        outlier_predictions = [self.detector.predict(outlier) for outlier in self.outlier]
        TP = sum(outlier_predictions)
        TN = len(self.inlier) - sum(inlier_predictions)
        FP = len(self.inlier) - TN
        FN = len(self.outlier) - TP
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        FPR = FP/(FP + TN)
        TPR = TP/(TP + FN)
        return accuracy, precision, recall, FPR, TPR

    def print_eval(self):
        accuracy, precision, recall, FPR, TPR = self.eval()
        print("accuracy", accuracy,
              "precision", precision,
              "recall", recall,
              "FPR", FPR,
              "TPR", TPR)

