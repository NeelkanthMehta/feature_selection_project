# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
import seaborn as sns

data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(df, size=11):
    ax = sns.heatmap(data.corr(), cmap='YlOrRd')
    plt.show()
