import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


df = pd.read_csv('assets/idx_dev.csv', header=None)

sns.countplot(y=1, data=df)
plt.ylabel('')
plt.title('Countplot on the training set')
plt.tight_layout()
plt.show()

########################################################

