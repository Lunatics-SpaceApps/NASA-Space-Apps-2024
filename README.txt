How to use the repository


What does the repository do?
It takes seismic data and returns the relative time of the quake.

Data preparation before model training
We needed to supply the model with a matrix containing all the velocity data of all the
seismic recording. This is done by going through each csv file, and saving the velocity
column. For the Moon test data, we faced a problem, as those csv were in different folders.
We decided to create a loop for accessing each folder in search for files. This process can 
be looped as many times as needed for accessing multiple subfolders.

This data is transposed and swapped with the zeros matrix. It’s important to note that x
dimensions are dynamic, as we check the quantity of files and the maximum number of
velocity data. As a result, many X matrix rows contain zeros, needed for providing the model
with a uniform data set.

In the case of the training data, we create a list containing each quake relative time. This is 
done utilizing a dedicated file we were provided with.

Challenges encountered
We count with little individual seismic data and a enormous amount of data for each. This is 
not optimal, as for the model the more seismic data the best, but also for the matrix creation.
Going through each file to discover the quantity of rows takes so much time for the little
quantity of files. A intermediate optimization could be reducing the resolution, eliminating half
the velocity data. This would have a negligible impact on the result and a great improvement
in processing times. 


NASA Wellcome

