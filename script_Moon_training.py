'''
2024/10/05 NASA International Space Apps (Leon, Spain)
By Lunatics
'''

import pandas as pd
import os
import numpy as np

def columns_rows(directory, csv_filenames):

    # Number of columns, number of detections
    num_columns = len(csv_filenames)

    # Variable to save the maximum number of rows, velocity data
    num_rows = 0

    #Current directory
    os.chdir(directory);

    # Go through all the files
    for file in csv_filenames:
        # Read each file
        df = pd.read_csv(file)
        # Count the number of rows of the current file
        rows = df.shape[0]

        # Compare with the earlier highest value, and save the maximum
        if rows > num_rows:
            num_rows =rows

    return num_columns, num_rows

def function_velocities_relativeTimes(directory1, directory2):
    # We take the detection files
    files = os.listdir(directory1)
    csv_filenames = [file for file in files if file.endswith(".csv")]

    num_columns, num_rows = columns_rows(directory1, csv_filenames)

   # Matrix for saving the velocities of all the detections
    x = np.zeros((num_columns,num_rows));
    i = 0;

    # We again save all the .csv file names
    csv_files = [file for file in os.listdir(directory1) if file.endswith('.csv')];

    # Iterates for every CSV file
    for file in csv_files:
        file_path = os.path.join(directory1, file);

        # Reads the CSV file
        df = pd.read_csv(file_path);

        # Assing the csv velocity column in a x row
        x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2]);
        i = i+1;

    # Y list saves the detection relative times
    os.chdir(directory2);
    df = pd.read_csv("apollo12_catalog_GradeA_final.csv");
    y = df.iloc[:,2];

    return x, y

directory1 = r'.\data\lunar\training\data\S12_GradeA'
directory2 = r'.\data\lunar\training\catalogs'

x, y = function_velocities_relativeTimes(directory1, directory2)

print(x)
print(y)
