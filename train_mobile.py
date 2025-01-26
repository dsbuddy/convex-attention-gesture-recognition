from cts import Data, MobileNetModel
from datetime import datetime,date


# gesture = 'tap'
gesture = 'swipe'

if gesture == 'tap':
    DATA_FILENAME = 'serial_MAC_NSEW_tap_ratio.json'
elif gesture == 'swipe':
    DATA_FILENAME = 'serial_MAC_upDownLeftRight_swipe_ratio.json'
else:
    exit(1)
batch = 'pad'
print('Loading trials...')
data = Data(DATA_FILENAME).load()
labels = data.labels()
training_set, test_set = data.split(.3)
print('Training model...')
modelName = 'mobilenet_{}_{}'.format(gesture, str(DATA_FILENAME.split('.')[0]))
model = MobileNetModel(labels, modelName)
model.train(training_set)
model.evaluate(training_set, test_set)
model.save_tflite()
model.evaluate_crossfold(data)