from loadData import download_data_from_S3, merge_with_local_data
from preprocessing import preprocessingForDNN
from train import train
from preprocessPrediction import loadNextMatch
import os
import sys


class PredictTask():

    def __init__(self, refresh_data, training):
        self.refreshData = refresh_data
        self.training = training

        if refresh_data == "True":
            os.system('source crawlAllTournaments.sh')
            download_data_from_S3()
            merge_with_local_data()
            preprocessingForDNN()

        if training == "True":
            train()

    def predict(self):

        loadNextMatch()


if __name__ == "__main__":
    refresh_data = sys.argv[1]
    train = sys.argv[2]
    predict = PredictTask(refresh_data=refresh_data, training=train)
    predict.predict()
