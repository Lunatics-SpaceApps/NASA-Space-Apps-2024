import pandas as pd
import os
import numpy as np

def columns_rows(directory, csv_filenames):

    # Number of columns, quantity of detection files
    num_columns = len(csv_filenames)

    # Variable to save the maximum number of rows, velocity detection data
    num_rows = 0
    # Selecting the current directory
    os.chdir(directory);

    # Go through all the files
    for file in csv_filenames:
        # Read each file
        df = pd.read_csv(file)
        # Count the number of rows of the current file, velocity data
        rows = df.shape[0]

        # Compare with the earlier highest value, and save the maximum
        if rows > num_rows:
            num_rows =rows

    return num_columns, num_rows

def function_velocities_relativeTimes(directory1, directory2):
    # Selecting the files in the current directory
    files = os.listdir(directory1)
    # Saving valuable files
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

        # Reads the CSV file
        df = pd.read_csv(file_path);

        # Velocity detecion values, a X row
        x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2]);
        i = i+1;

    # Y list relative detecion time
    os.chdir(directory2);
    df = pd.read_csv("Mars_InSight_training_catalog_final.csv");
    y = df.iloc[:,2];

    return x, y

directory1 = r'.\data\mars\training\data'
directory2 = r'.\data\mars\training\catalogs'

x, y = function_velocities_relativeTimes(directory1, directory2)

print(x)
print(y)
