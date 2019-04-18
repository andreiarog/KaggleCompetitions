#imports
import pandas as pd
import numpy as np
import functions as func #file containing the user created functions

#read csv files and create dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train)

#comments to Andreia
#Name field might be relevant when NaN (convert to binary 'known_name') - higher probability of death or euthanasia (?) due to poorer condition when arriving at shelter
#0 probability of return_to_owner (? check no. NaN name observations per value of outcome)
#check dog/cat balance
	
#main
#.apply allows to apply functions to each row or column

df = func.pre_process(train)

#some dummies still missing
#need to drop AnimalID or just exclude it in models?
print(df)
df.to_csv('dataframe.csv') #are we happy with the final dataframe?

#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good correlation between sex_intervention and outcome	
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min
#can't find a good size and intelligence list for cats; breed info will be very different based on animal type; should classify separately?
#size and intelligence list for dogs still incomplete, but covering most frequent breeds (better for size)


#unbalanced attribute, with Black and Black/White having considerably more observations than other colours




