'''
2024/10/05 NASA International Space Apps (Leon, Spain)
By Lunatics
'''

import pandas as pd
import os
import numpy as np

# For obtaining the quantity of detections and maximum velocity data
def columns_rows(directory):

    num_rows = 0
    num_columns = 0
    rows = 0
    # We go through each available folder, and assign the directory
    for folder in os.listdir(directory):
        new_directory = os.path.join(directory, folder)

        # We test if it is in fact a folder
        if os.path.isdir(new_directory):

            os.chdir(new_directory)
            # We get the file names list of the folder
            files = os.listdir(new_directory)

            # We obtain the quantity of files
            csv_filenames = [file for file in files if file.endswith(".csv")]

            # Go through the files
            for file in csv_filenames:
                # Read each file
                df = pd.read_csv(file)
                # Count the number of rows of the current file, velocity data
                rows = df.shape[0]

                # Compare with the earlier highest value, and save the maximum
                if rows > num_rows:
                    num_rows =rows

        # Go back to the parent folder
        os.chdir(directory)

        # We add the quantity of files, in order to obtain the total ammount
        num_columns = num_columns + len(csv_filenames)

    return num_columns, num_rows

# Obtaining the Matrix for all the velocities and detections
def function_velocities_relativeTimes(directory):

    num_columns, num_rows = columns_rows(directory)

    # num_columns is the number of columns of the CSV, velocity data
    # num_rows is the quantity of csv files
    x = np.zeros((num_columns, num_rows))
    i = 0

    # We go through each folder
    for folder in os.listdir(directory):
        new_directory = os.path.join(directory, folder)

        # Checking if it is in fact a folder
        if os.path.isdir(new_directory):

            # Changing the directory for the current one.
            os.chdir(new_directory)

            csv_files = [file for file in os.listdir(new_directory) if file.endswith('.csv')]

            # Iterates for every CSV file
            for file in csv_files:
                file_path = os.path.join(new_directory, file)

                # Reads the CSV file
                df = pd.read_csv(file_path)

                # Take the velocities of each detection, and assing them an x row.
                x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2])
                i = i+1

            # Back to the parent folder
            os.chdir(directory)

    return x

directory = r'.\data\lunar\test\data'

x = function_velocities_relativeTimes(directory)

print(x)
