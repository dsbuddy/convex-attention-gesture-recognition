from data import Data
from model import PadModel
from datetime import datetime,date

DATA_FILENAME = 'data_sample.json'

print('Loading trials...')
data = Data(DATA_FILENAME).load()

labels = data.labels()
training_set, test_set = data.split(.3)

print('Training model...')
modelName = DATA_FILENAME.split('.')[0]
model = PadModel(labels, modelName)
#model.train(training_set)

print('Evaluating model...')
#model.evaluate(training_set, test_set)

model.cleanUpModels()

