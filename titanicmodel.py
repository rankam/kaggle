from titanicdata import female
from titanicdata import female_test
from titanicdata import male_test
from titanicdata import male
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
from sklearn import svm, grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

# dictionary to create dataframe headers
traindic = {0:'survived',1:'class',2:'sex',3:'age',4:'siblings',5:'parents',6:'fare',7:'embarked'}
testdic = {0:'class',1:'sex',2:'age',3:'siblings',4:'parents',5:'fare',6:'embarked'}

# putting dataframe headers on dataframes
female = pd.DataFrame(female)
female_test = pd.DataFrame(female_test)
male = pd.DataFrame(male)
male_test = pd.DataFrame(male_test)
female= female.rename(columns=traindic)
female_test= female_test.rename(columns=testdic)
male= male.rename(columns=traindic)
male_test= male_test.rename(columns=testdic)

# separates 1st,2nd,3rd class passengers into binary bins
def class_number(dataset,feature):
  class1=[]
	class2=[]
	class3=[]
	for x in dataset[feature]:
		if x ==1:
			class1.append(1)
		else:
			class1.append(0)
		if x==2:
			class2.append(1)
		else:
			class2.append(0)
		if x==3:
			class3.append(1)
		else:
			class3.append(0)
	dataset['class1']=class1
	dataset['class2']=class2
	dataset['class3']=class3
	dataset.pop(feature)

# combines 'has parents' and 'has children' into binary bin 'has family'
def family(dataset,feature,feature2):
	family_present=[]
	for x,y in zip(dataset[feature],dataset[feature2]):
		if x > 0 or y > 0:
			family_present.append(1)
		else:
			family_present.append(0)
	dataset['family_present']= family_present
	dataset.pop(feature)
	dataset.pop(feature2)

# apply log transformation
famlist=[female,male,female_test,male_test]
def logfeature(dataset,feature):
	dataset[feature]=np.log(1+dataset[feature])

for fam in famlist:
	logfeature(fam,'fare')

# Create bins for ticket fare
ten = []
twenty = []
thirty = []
abovethirty=[]
def farebin(dataset,feature,int1,int2,listname,newlist):
    listname=[]
    for x in dataset[feature]:
        if x <np.log(int1) and x >np.log(int2):
            listname.append(1)
        else:
            listname.append(0)
    dataset[newlist]=listname
    if 'abovethirty' in dataset:
    	dataset.pop(feature)

# apply bins to datasets   	
for fam in famlist:
	farebin(fam,'fare',10,0,ten,'ten')

for fam in famlist:
	farebin(fam,'fare',20,10,twenty,'twenty')

for fam in famlist:
	farebin(fam,'fare',30,20,thirty,'abovetwenty')
for fam in famlist:
	farebin(fam,'fare',300000,30,abovethirty,'abovethirty')

# create bins for ages
# NOT USING AGE IN CURRENT MODEL
# agebelowten=[]
# agebelowtwenty=[]
# agebelowforty=[]
# ageaboveforty=[]

# def agebin(dataset,feature,int1,int2,listname,newlist):
#     listname=[]
#     for x in dataset[feature]:
#         if x <int1 and x >int2:
#             listname.append(1)
#         else:
#             listname.append(0)
#     dataset[newlist]=listname
#     if 'ageaboveforty' in dataset:
#     	dataset.pop(feature)

# for fam in famlist:
# 	agebin(fam,'age',10,0,agebelowten,'ageten')

# for fam in famlist:
# 	agebin(fam,'age',20,10,agebelowtwenty,'agetwenty')

# for fam in famlist:
# 	agebin(fam,'age',40,20,agebelowforty,'ageforty')

# for fam in famlist:
# 	agebin(fam,'age',40000,40,ageaboveforty,'ageaboveforty')

for fam in famlist:
    family(fam,'siblings','parents')

for fam in famlist:
	class_number(fam,'class')

# update datasets to reflect above changes to categories
female,male,female_test,male_test = famlist[0],famlist[1],famlist[2],famlist[3]

# add id number
female['idnumber']=range(len(female['class1']))
female_test['idnumber']=range(len(female_test['class1']))
male['idnumber']=range(len(male['class1']))
male_test['idnumber']=range(len(male_test['class1']))

# remove age catagory 
female.pop('age')
female_test.pop('age')
male.pop('age')
male_test.pop('age')

#convert to numpy array
femaletrain=np.array(female)
femaletest=np.array(female_test)

#create training sets
trainx=femaletrain[:,1:-1]
trainy=femaletrain[:,0]
testx=femaletest[:,:-1]
maletrain=np.array(male)
maletest=np.array(male_test)


# np.random.shuffle(maletrain)
# np.random.shuffle(maletest)
# np.random.shuffle(femaletrain)
# np.random.shuffle(femaletest)

# fit the model
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.1, max_features=None, compute_importances=False, random_state=None).fit(trainx,trainy)
predictions = clf.predict(testx)
# print 'n_estimators = ',est,'learning_rate = ',learn
# 			print dep
print np.mean(cross_val_score(clf,trainx,trainy))

# for pred,idn in zip(predictions,female_test['idnumber']):
# 	print pred,',',idn

trainx=maletrain[:,1:-1]
trainy=maletrain[:,0]
testx=maletest[:,:-1]

# fit the model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_density=0.1, max_features=None, compute_importances=False, random_state=None).fit(trainx,trainy)
predictions = clf.predict(testx)
print np.mean(cross_val_score(clf,trainx,trainy))
