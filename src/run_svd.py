import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.CONSTANTS import DATA_PATH
from utils.calculate_metrics import binary_classification_metrics, regression_metrics


## Load Processed Data
train_data = pd.read_csv(DATA_PATH + '/train_data_it.csv')
test_data = pd.read_csv(DATA_PATH + '/test_data_it.csv')

## Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader).build_full_trainset()

## Train SVD (Singular Value Decomposition) model
model = SVD()
model.fit(trainset)

## Make predictions on the test set
testset = list(test_data[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
predictions = [model.predict(uid, iid).est for uid, iid, _ in testset]
actuals = [rating for _, _, rating in testset]

## Compute regression metrics 
regression_metrics(actuals, predictions)

## Compute classification metrics
binary_classification_metrics(actuals, predictions)

    