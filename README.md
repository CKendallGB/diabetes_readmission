# Diabetes readmission

The purpose of this project is to examine data on diabetes patients in the US and develop code that determines whether a patient is likely to be readmitted or not. The data for the project can be found in this github repository and at:

https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

along with another csv file explaining the meanings of the various id columns.  
The explanation of the different fields can be found at:

https://www.hindawi.com/journals/bmri/2014/781670/tab1/

The code is structured as a series of functions all executed at the end of the code. It may be worthwhile looking down there first to see in what order things are executed and to then read the functions (though the functions are defined in order of use). Outputs (graphs, printed text) are normally saved in the folder this program is located in due to the number of them and the occasional difficulty in viewing them. Typically there will be a number of graphs (normally) named after columns and a txt file communicating more numerical information.

A large amount of the data dealt with was categorical data, to clarify how it's talked about I will be calling the overall information a 'field' which has different 'categories'. For example age is a field with the categories 10-20, 20-30, etc. In most circumstances a field will be interchangeable with a column until the data is one hot encoded, at which point a field will span as many columns as is has categories.

Typical running time is around 3-5 minutes without optimising parameters.
