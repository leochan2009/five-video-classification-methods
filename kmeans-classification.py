"""
Train our RNN on extracted features or images.
"""
from models import ResearchModels
from data import DataSet
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = DataSet(
        seq_length=30,
        class_limit=4, labelEncoding='coral_ordinal')

    X, y = data.get_all_sequences_in_memory('train', 'features')
    X_mean = np.average(X,axis=1)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_mean)
    y_predict = kmeans.predict(X_mean)
    print(y_predict, y)
    plt.plot(y_predict, label="modeled")
    plt.plot(y + 0.05, label='accessed')
    plt.legend()
    plt.show()

    classes = ['clarity 25','clarity 50','clarity 75','clarity 100']
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(221)
    for index, ele in enumerate(classes):
        ax1.scatter(X_mean[y==index, 0], X_mean[y==index, 1], label=ele,
                   alpha=0.3, edgecolors='none')
        ax1.set_xlabel('EntR')
        ax1.set_ylabel('EntG')
    ax1.legend()
    ax2 = fig.add_subplot(222)
    for index, ele in enumerate(classes):
        ax2.scatter(X_mean[y == index, 2], X_mean[y == index, 3], label=ele,
                   alpha=0.3, edgecolors='none')
        ax2.set_xlabel('EntB')
        ax2.set_ylabel('hiFreq')
    ax2.legend()
    ax3 = fig.add_subplot(223)
    for index, ele in enumerate(classes):
        ax3.scatter(X_mean[y == index, 6], X_mean[y == index, 7], label=ele,
                    alpha=0.3, edgecolors='none')
        ax3.set_xlabel('SumG')
        ax3.set_ylabel('SumR')
    ax3.legend()
    ax4 = fig.add_subplot(224)
    for index, ele in enumerate(classes):
        ax4.scatter(X_mean[y == index, 8], X_mean[y == index, 10], label=ele,
                    alpha=0.3, edgecolors='none')
        ax4.set_xlabel('SumB')
        ax4.set_ylabel('RedRatio')
    ax4.legend()
    plt.show()

if __name__ == '__main__':
    main()
