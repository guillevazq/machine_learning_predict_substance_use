import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('student_data/student-mat.csv')

# In order to see the correlation between the weekday, and weekend alcohol use
def graph_comparaison_weekend_weekdays(students=30):
    width = 0.33333
    
    x = np.arange(students)
    y1 = data['Dalc'].values[:students]
    y2 = data['Walc'].values[:students]

    plt.bar(x - width, y1, label='Daily', width=width)
    plt.bar(x + width, y2, label='Weekends', width=width)

    plt.legend()
    plt.show()

# See the correlation of each column with the weekend consumption
correlation_weekend_columns = data.corr()['Walc'].sort_values(ascending=False)