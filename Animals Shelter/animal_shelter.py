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
#.apply allows to apply functions to each row or column

#sex
df = func.column_split(train, 'SexuponOutcome', ' ', 2, ['sex_intervention', 'sex'])
df['sex']= df['sex'].apply(func.unknown_value,1)
df['sex_intervention']= df['sex_intervention'].apply(func.unknown_value,1).apply(func.sex_int,1)
#print(pd.pivot_table(df,index=['sex','sex_intervention'], columns = ['OutcomeType'], aggfunc = 'count'))

df = func.create_dummies(df,'sex')
df = func.create_dummies(df,'sex_intervention')
#drop 3rd dummy since perfectly correlated with others?

#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good correlation between sex_intervention and outcome	

#age
df['age'] = df['AgeuponOutcome'].apply(func.age_standard,1)
df['age_known']=df['age'].apply(func.unknown_age,1)
#0 days consider unknown age (? can't remember what we agreed)

print(pd.pivot_table(df,index=['age_known'], columns = ['OutcomeType'], aggfunc = 'count'))
#there are very few animals with unknown age. Exclude those with unknown? Or replace age by mean? Which mean (overall, class, etc)?

print(pd.pivot_table(df,values=['age'],columns = ['OutcomeType'], aggfunc = [min,max,np.mean]))
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min

#Breed

df['pure_mix'] = df['Breed'].apply(func.pure_mix,1)
df['dangerous'] = df['Breed'].apply(func.dangerous_breed,1)
df['size'] = df['Breed'].apply(func.breed_size,1)
df['intelligence'] = df['Breed'].apply(func.breed_intelligence,1)
print(pd.pivot_table(df,index=['intelligence'], columns = ['AnimalType'], aggfunc = 'count'))

#can't find a good size and intelligence list for cats; breed info will be very different based on animal type; should classify separately?
#should we do a var for hair length for cats? Hypoallergenic breeds?
#size and intelligence list for dogs still incomplete, but covering most frequent breeds (better for size)



#some breeds have very few observations, to use this attribute a more summarised one will be needed (size, country of origin, life expectancy?)

#Colour
#Specified when 1 or 2, more than 2 as 'tricolored'; worth creating a no. colours variable? Doesn't seem useful
#pd.pivot_table(df,index=['Color'], columns = ['OutcomeType'], aggfunc = 'count').to_csv('colour_pivot.csv')
#unbalanced attribute, with Black and Black/White having considerably more observations than other colours
#only use I can envision for colour would be for likelihood of adoption, with some colours being more likely to be adopted, but can't see any trend

#convert all categorical variables to dummy using function already created

#'pre_processing' function that applies all final changes that we decide and then apply this to both train and test
df.to_csv('dataframe.csv')
#print(df)

