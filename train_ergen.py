from cts import Data, PyTorchErgenModel

gesture = 'tap'
# gesture = 'swipe'

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
modelName = 'ergen_{}_{}'.format(gesture, str(DATA_FILENAME.split('.')[0]))
model = PyTorchErgenModel(labels, modelName)
model.train(training_set)
model.evaluate_on_training_and_test(training_set, test_set) 
model.evaluate_with_crossfold(data)