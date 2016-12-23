import numpy as np
from sklearn import svm
import sklearn.preprocessing
from aa import *
import matplotlib.pyplot as plt


def find_penalties(cs=range(-10, 11)):
    for i in cs:
        lsvc = svm.LinearSVC(C=10**i, loss="hinge", fit_intercept=False)
        lsvc.fit(train_data, train_labels)
        yield i, lsvc.score(validation_data, validation_labels), lsvc.score(train_data, train_labels)


def halving():
    a, b = -10, 10
    diff = b - a
    best_c = 0

    for i in range(21):
        scores = list(find_penalties(numpy.linspace(a, b, 21)))
        c = max(scores, key=lambda x: x[1])[0]
        if c == best_c:
            return
        yield scores, a, b, c

        best_c = c
        b = c + diff / 4
        a = c - diff / 4
        diff = diff / 2


halving_scores = list(halving())


for scores, a, b, c in halving_scores:
    print c
    plt.plot(numpy.linspace(a, b, 21), [y for (x, y, z) in scores])
    plt.plot(numpy.linspace(a, b, 21), [z for (x, y, z) in scores])

plt.ylabel('Scores')
plt.xlabel('C')
plt.show()
