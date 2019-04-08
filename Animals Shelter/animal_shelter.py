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

#sex
df = func.column_split(train, 'SexuponOutcome', ' ', 2, ['sex_intervention', 'sex'])
df['sex']= df['sex'].apply(func.unknown_value,1)
df['sex_intervention']= df['sex_intervention'].apply(func.unknown_value,1).apply(func.sex_int,1)
#print(pd.pivot_table(df,index=['sex','sex_intervention'], columns = ['OutcomeType'], aggfunc = 'count'))

df = func.create_dummies(df,'sex')
df = func.create_dummies(df,'sex_intervention')

#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good correlation between sex_intervention and outcome	

#age
df['age'] = df['AgeuponOutcome'].apply(func.age_standard,1)
df['age_known']=df['age'].apply(func.unknown_age,1)
#replace unknown by mean per animal type 
df_trans = (df.groupby('AnimalType')).transform(lambda x: x.fillna(round(x.mean()), inplace = False)) #not working, read documentation to understand better
df['age'] = df_trans['age']
df['age_bins']= df['age'].apply(func.age_bins,1)

print(pd.pivot_table(df,values=['age'],columns = ['OutcomeType'], aggfunc = [min,max,np.mean]))
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min

#Breed

df['pure_mix'] = df['Breed'].apply(func.pure_mix,1)
df['dangerous'] = df['Breed'].apply(func.dangerous_breed,1)
df['size'] = df['Breed'].apply(func.breed_size,1)
df['intelligence'] = df['Breed'].apply(func.breed_intelligence,1)
df['hypoaller'] = df['Breed'].apply(func.breed_hypoaller,1)

#print(pd.pivot_table(df,index=['intelligence'], columns = ['AnimalType'], aggfunc = 'count'))

#can't find a good size and intelligence list for cats; breed info will be very different based on animal type; should classify separately?
#should we do a var for hair length for cats? Hypoallergenic breeds?
#size and intelligence list for dogs still incomplete, but covering most frequent breeds (better for size)

#Colour

df['Colour'] = df['Color'].apply(func.process_colour,1)
df['no_colours'] = df['Colour'].apply(func.breed_colour,1)
#print(pd.pivot_table(df,index=['no_colours'], columns = ['AnimalType'], aggfunc = 'count'))

df = func.create_dummies(df,'no_colours')

df['common_colour'] = func.common_value(df['Colour'],95) #considers 16 most common colours as common (more than 550 occurrences)
#build different vars for different percentile values (90,92,95,98)

#unbalanced attribute, with Black and Black/White having considerably more observations than other colours

#convert all categorical variables to dummy using function already created - eliminate one of the dummies generated for all variables

#'pre_processing' function that applies all final changes that we decide and then apply this to both train and test
df.to_csv('dataframe.csv')
#print(df)

