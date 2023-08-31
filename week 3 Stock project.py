import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import csv

if __name__ == "__main__":
    #defining the file path to the csv data set
    file_path = input("Enter the path to the CSV file: ")
    data = []
    #opening and reading the file
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #iterates through each row in the file and appends it to the empty matrix
        for row in csv_reader:
            data.append(row)
        #transferes the now full matrix to a numpy array
        npdata = np.array(data)

    #sliced the numpy array by collomn for specific a specific data set
    unfiltered_dataset = npdata[:, 7:]
    #filtering out any empty rows that could have been appended
    non_empty_rows = np.any(unfiltered_dataset != '', axis=1)
    filtered_data = unfiltered_dataset[non_empty_rows]

    #declaring which variable will be indpendent and dependent variables
    x = filtered_data['Cigarettes smoked per day by males']
    y = filtered_data['Total Cholesterol levels']

    #creating the scatter plot and labels
    plt.scatter(x,y)
    plt.xlabel('Cigarettes smoked per day by males')
    plt.ylabel('Total Cholesterol levels')

    #creating line of best fit with Linear Regression
    model = LinearRegression(fit_intercept = True).fit(x, y)
    xfit = np.linspace(0, 50)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.plot(xfit,yfit, color = 'r')
    plt.show()

    #Calculating R-Squared
    r_squared = model.score(x,y)
    print('R-Squared:', r_squared)