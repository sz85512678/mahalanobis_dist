import pandas as pd


class Data:
    """
    Hold time-series data and allow augmentations
    """

    def __init__(self):
        self.corpus = None  # unlabelled corpus consists of streams, numpy.array(numpy.array)
        self.test_inlier = None  # test set consists of numpy.array of inliers,
        self.test_outlier = None  # test set consists of numpy.array of outliers

    def get_corpus(self):
        return self.corpus

    def get_test_in(self):
        return self.test_inlier

    def get_test_out(self):
        return self.test_outlier

    def load_pen_digit(self, digit: int, path_to_pickle: str):
        """
        Load pen digit dataset with a specific digit as training set
        :param digit: 0-9, use as "normality" training corpus
        :param path_to_pickle: path to directory containing pickle file for train and test
        :return: None
        """
        train_df = pd.read_pickle(path_to_pickle + "pen_digit_train.pkl")
        test_df = pd.read_pickle(path_to_pickle + "pen_digit_test.pkl")
        self.corpus = train_df[train_df["Digit"] == digit]["Stream"].values
        self.test_inlier = test_df[test_df["Digit"] == digit]["Stream"].values
        self.test_outlier = test_df[test_df["Digit"] != digit]["Stream"].values

