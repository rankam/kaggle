import csv as csv 
import numpy as np

csv_file_object = csv.reader(open('/users/aaronrank/datascience/kaggle/titanic/data/train.csv', 'rb')) 
header = csv_file_object.next()  
                                
data=[]                          
for row in csv_file_object:      
    data.append(row)             
data = np.array(data)           

csv_file_object = csv.reader(open('/users/aaronrank/datascience/kaggle/titanic/data/test.csv', 'rb')) 
header = csv_file_object.next()   
                                 
testdata=[]                          
for row in csv_file_object:      
    testdata.append(row)
test_data = np.array(testdata) 


# load training data
titanic = data[:,(0,1,3,4,5,6,8,10)]

# Pre-Processing
titanic[::,2][titanic[::,2]=='male'] = 0
titanic[::,2][titanic[::,2]=='female'] = 1
titanic[:,-1][titanic[:,-1]=='S']=0
titanic[:,-1][titanic[:,-1]=='Q']=1
titanic[:,-1][titanic[:,-1]=='C']=2

# Split training set by male and female
female = titanic[titanic[:,2]=='1']
male = titanic[titanic[:,2]=='0']

# Replace missing values with averages 
male[:,3][male[:,3]=='']=30.7
female[:,3][female[:,3]=='']=27.9
female[:,-1][female[:,-1]=='']=0

# Convert all values to floats
female=female.astype(np.float)
male=male.astype(np.float)

# Scikit-Learn Preprocessing of catagorical data



# load test data
titanic_test = test_data[:,(0,2,3,4,5,7,9)]

# Pre-processing
titanic_test[::,1][titanic_test[::,1]=='male'] = 0
titanic_test[:,1][titanic_test[::,1]=='male'] = 0
titanic_test[:,1][titanic_test[::,1]=='female'] = 1
titanic_test[:,-1][titanic_test[:,-1]=='S']=0
titanic_test[:,-1][titanic_test[:,-1]=='Q']=1
titanic_test[:,-1][titanic_test[:,-1]=='C']=2


# Split test set by male and female
female_test = titanic_test[titanic_test[:,1]=='1']
male_test = titanic_test[titanic_test[:,1]=='0']

# Replace missing values with averages
male_test[:,2][male_test[:,2]=='']=30.2
male_test[:,-2][male_test[:,-2]=='']=26.7
female_test[:,2][female_test[:,2]=='']=30.2
female_test[:,-1][female_test[:,-1]=='']=0


# Split data into male and female sets
female_test=female_test.astype(np.float)
male_test=male_test.astype(np.float)
