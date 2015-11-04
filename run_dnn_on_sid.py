__author__ = 'zhangxulong'
from numpy import *
import dnn
import time
from datasets import load_imdb
from pybrain.utilities import percentError

""" Mean Root Squared Error
Accepts a list of predictions and a list of targets """


def mrse(preds, targets):
    dif = [(math.sqrt((p - targets[i]) ** 2)) for i, p in enumerate(preds)]
    score = float(sum(dif)) / len(targets)
    return score


""" Root Mean Squared Error
Accepts a list of predictions and a list of targets """


def rmse(preds, targets):
    dif = [((p - targets[i]) ** 2) for i, p in enumerate(preds)]
    mean = float(sum(dif)) / len(targets)
    root = math.sqrt(mean)
    return root


# time_start
timestart = time.time()

# trainset, testset = split_dataset(dataset) #it need shuffle##TODO
trainset, testset = load_imdb('datasets/dataset.pkl')


# train
train_data = trainset[0]
train_target = trainset[1]
class_num = len(unique(train_target))
print ("class num:", class_num)

# set the layers parm,and the max is 8 now
layers = [13, 20, 10, 5, class_num]  # first 10 is the 95mfcc's first 10 dim
# run the dnn ,first autoencoder and then DNNRegressor
autoencoder = dnn.AutoEncoder(train_data, train_data, train_target, layers, hidden_layer="SigmoidLayer",
                              final_layer="SoftmaxLayer", compression_epochs=1, bias=True, autoencoding_only=False,
                              dropout_on=True)
print("1here is okay")  # ============================================================
autoencoder.fit()
# time end
timeend = time.time()
# train time
traintime = timeend - timestart
print"====================train the dnn take %d second ==========================" % traintime
# test
test_time_start = time.time()

test_target = testset[1]
predict_target = []
for test_data in testset[0]:
    print_value = 10
    if print_value > 0:
        print "test_data:%s=============" % str(test_data)
        print "predict:%s===============" % str(autoencoder.predict(test_data))
    print_value -= 1
    predict_target.append(argmax(autoencoder.predict(test_data)))
test_result = 100 - percentError(predict_target, test_target)
mrse_result = mrse(predict_target, test_target)
rmse_result = rmse(predict_target, test_target)
print "====================mrse preds {0}======================".format(mrse_result)
print "====================rmse preds {0}======================".format(rmse_result)
print "=================the true target is below==================="
print test_target[0:50]
print "------------------------------------------------------------"
print predict_target[0:50]
print "=================the out target is upon==================="
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
print "the test acc result is %.2f%%" % test_result
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
test_time_end = time.time()
test_time = test_time_end - test_time_start
print"====================make the test take %d second ==========================" % test_time
