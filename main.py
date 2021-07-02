import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

student_data = pd.read_csv('student_student_data/student-mat.csv')

# Separate the test set and the training set

# In order to see the correlation between the weekday, and weekend alcohol use
def graph_comparaison_weekend_weekdays(students=30):
    width = 0.33333
    
    x = np.arange(students)
    y1 = student_data['Dalc'].values[:students]
    y2 = student_data['Walc'].values[:students]

    plt.bar(x - width, y1, label='Daily', width=width)
    plt.bar(x + width, y2, label='Weekends', width=width)

    plt.legend()
    plt.show()

# See the correlation of each column with the weekend consumption
correlation_weekend_columns = student_data.corr()['Walc'].sort_values(ascending=False)

# Remove the daily 