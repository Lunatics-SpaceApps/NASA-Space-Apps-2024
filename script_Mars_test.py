import pandas as pd
import os
import numpy as np

def columns_rows(directory, csv_filenames):

    # Number of columns, quantity of detection files
    num_columns = len(csv_filenames)

    # Variable to save the maximum number of rows, velocity data
    num_rows = 0
    # Selecting the current directory
    os.chdir(directory);

    # Go through the files
    for file in csv_filenames:
        # Read each file
        df = pd.read_csv(file)
        # Count the number of rows of the current file
        rows = df.shape[0]

        # Compare with the earlier highest value, and save the maximum
        if rows > num_rows:
            num_rows =rows

    return num_columns, num_rows

def function_velocities_relativeTimes(directory1):

    files = os.listdir(directory1)
    csv_filenames = [file for file in files if file.endswith(".csv")]

    num_columns, num_rows = columns_rows(directory1, csv_filenames)

    # X Matrix, for saving all the velocity data for each detection
    x = np.zeros((num_columns,num_rows));
    i = 0;

    # Taking the valuable files
    csv_files = [file for file in os.listdir(directory1) if file.endswith('.csv')];

    # Iterates for every CSV file
    for file in csv_files:
        file_path = os.path.join(directory1, file);

        # Reads thee CSV file
        df = pd.read_csv(file_path);

        # Velocity dat assigned to a x row
        x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2]);
        i = i+1;

    return x

directory1 = r'.\data\mars\test\data'

x=function_velocities_relativeTimes(directory1)

print(x)
