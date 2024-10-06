import pandas as pd
import os
import numpy as np

def colums_rows(directory, csv_filenames):

    # Number of colums
    num_colums = len(csv_filenames)

    # Variable to save the maximum number of rows
    num_rows = 0
    
    os.chdir(directory)
    
    # Go through the files
    for file in csv_filenames:
        # Read each file
        df = pd.read_csv(file)
        # Count the number of rows of the current file
        rows = df.shape[0]

        # Compare with the earlier highest value, and save the maximum
        if rows > num_rows:
            num_rows =rows

    return num_colums, num_rows

def training_data(directory1, directory2):

    '''
    files = os.listdir(directory1)
    csv_filenames = [file for file in files if file.endswith(".csv")]

    num_colums, num_rows = colums_rows(directory1, csv_filenames)'''
   
    x = np.zeros((76,572427))
    i = 0

    # MATRIX x (velocities)
    csv_files = [file for file in os.listdir(directory1) if file.endswith('.csv')]

    # Iterates for every CSV file
    for file in csv_files:
        file_path = os.path.join(directory1, file)

        # Reads thee CSV file
        df = pd.read_csv(file_path)

        # Discard everything except the velocity column and place it in the rows
        x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2])
        i = i+1
        
    # VECTOR y (relative times)
    os.chdir(directory2)
    df = pd.read_csv("apollo12_catalog_GradeA_final.csv")
    y = df.iloc[:,2]

    return x, y

def test_data(directory1):

    files = os.listdir(directory1)
    csv_filenames = [file for file in files if file.endswith(".csv")]

    num_colums, num_rows = colums_rows(directory1, csv_filenames)

    x = np.zeros((num_colums, num_rows))
    i = 0

    # MATRIX x (velocities)
    csv_files = [file for file in os.listdir(directory1) if file.endswith('.csv')]

    # Iterates for every CSV file
    for file in csv_files:
        file_path = os.path.join(directory1, file)

        # Reads thee CSV file
        df = pd.read_csv(file_path)

        # Discard everything except the velocity column and place it in the rows
        x[i,0:len(df.iloc[:,2])] = np.transpose(df.iloc[:,2])
        i = i+1

    return x

#directory1 = "data/lunar/training/data/S12_GradeA"
#directory2 = "data/lunar/training/catalogs"

#x, y = function_velocities_relativeTimes(directory1, directory2)

#print(x)
#print(y)
