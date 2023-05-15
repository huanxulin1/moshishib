from dataset import dataset
from model import KNN_train

if __name__ == '__main__':
    data_file = "Pistachio_28_Features_Dataset.xlsx"
    X_train, Y_train, X_test, Y_test = dataset(data_file)
    KNN_train(X_train, Y_train, X_test, Y_test, [1, 2, 5, 8, 10, 15, 19])