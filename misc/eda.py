import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

sns.set_style('darkgrid')

########################################################

def countplot():
    df = pd.read_csv('assets/idx_dev.csv', header=None)
    sns.countplot(y=1, data=df)
    plt.ylabel('')
    plt.title('Countplot on the training set')
    plt.tight_layout()
    plt.show()

########################################################




if __name__ == "__main__":

    countplot()
