#imports
import pandas as pd
import numpy as np
import functions as func #file containing the user created functions

#read csv files and create dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
	
#main
#.apply allows to apply functions to each row or column

df=func.pre_process(train)
df.to_csv('dataframe.csv') 
features_select = func.feature_selection(df.drop(['AnimalID'],axis=1),'OutcomeType')
print(features_select)


#feature selection comments:
#age appears above all age bins except 1, even though in general age bins are good features, use age?
#hour sin and cos have good ranking, how to use them?

#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good correlation between sex_intervention and outcome	
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min
#can't find a good size and intelligence list for cats; breed info will be very different based on animal type; should classify separately?
#size and intelligence list for dogs still incomplete, but covering most frequent breeds (better for size)


#unbalanced attribute, with Black and Black/White having considerably more observations than other colours




