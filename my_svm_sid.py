from datasets import load_imdb
from keras.preprocessing import sequence

__author__ = 'zhangxulong'
from sklearn import svm

max_features = 20000
maxlen = 100
print("Loading data...")
(X_train, y_train), (X_test, y_test) = load_imdb(imdb_dataset='datasets/6singers_13mfcc_nolink.pkl',
                                                 nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
x = X_train
y = y_train
print('Build model...')
modle = svm.SVC()
modle.fit(x, y)
predict_result = modle.predict(X_test)
accurate = 0.
num = 0.
total = len(y_test)
for i in range(len(y_test)):
    if predict_result[i] == y_test[i]:
        num += 1
    else:
        num += 0
accurate = num / total
print accurate