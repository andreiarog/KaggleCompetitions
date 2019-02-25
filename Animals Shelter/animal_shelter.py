#imports
import pandas as pd
import numpy as np
import functions as func #file contained the user created functions

#read csv files and create dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train)

#comments to Andreia
#Name field might be relevant when NaN (convert to binary 'known_name') - higher probability of death or euthanasia (?) due to poorer condition when arriving at shelter
#0 probability of return_to_owner (? check no. NaN name observations per value of outcome)
#check dog/cat balance
	
#main	
df = func.column_split(train, 'SexuponOutcome', ' ', 2, ['sex_intervention', 'sex'])
df['sex']= df['sex'].apply(func.unknown_value,1)
df['sex_intervention']= df['sex_intervention'].apply(func.unknown_value,1)

#.apply allows to apply functions to each row or column
df['age'] = df['AgeuponOutcome'].apply(func.age_standard,1)
#0 years = 0.0 and blanks kept as nulls. Convert all to the same way of treating nulls? Which? Or what does 0 years mean?

df['pure_mix'] = df['Breed'].apply(func.pure_mix,1)

df['dangerous'] = df['Breed'].apply(func.dangerous_breed,1)

#for all attributes look at no. observations per attribute_value vs outcome_value 
#age
print(pd.pivot_table(df,values=['age'],columns = ['OutcomeType'], aggfunc = [min,max,np.mean]))
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min

#sex
print(pd.pivot_table(df,index=['sex','sex_intervention'], columns = ['OutcomeType'], aggfunc = 'count'))
#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good coreelation between sex_intervention and outcome

#Breed
#pd.pivot_table(df,index=['Breed'], columns = ['OutcomeType'], aggfunc = 'count').to_csv('breed_pivot.csv')
#what else can be done with breed?
#cats are less likely to be returned to owner ?
#some breeds have very few observations, to use this attribute a more summarised one will be needed (size, country of origin, life expectancy?)

#Colour
#Specified when 1 or 2, more than 2 as 'tricolored'; worth creating a no. colours variable? Doesn't seem useful
#pd.pivot_table(df,index=['Color'], columns = ['OutcomeType'], aggfunc = 'count').to_csv('colour_pivot.csv')
#unbalanced attribute, with Black and Black/White having considerably more observations than other colours
#only use I can envision for colour would be for likelihood of adoption, with some colours being more likely to be adopted, but can't see any trend

#convert all categorical variables to dummy

#'pre_processing' function that applies all final changes that we decide and then apply this to both train and test
#df.to_csv('dataframe.csv')

