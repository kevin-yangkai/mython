from sklearn import svm

x = []
y = []
k=0.
for i in range(9):
    for j in range(9):
        k = i + j
        x.append([i, j])
        y.append(k)
modle = svm.SVR()
x_y = modle.fit(x, y)
print modle.predict([8, 8])

