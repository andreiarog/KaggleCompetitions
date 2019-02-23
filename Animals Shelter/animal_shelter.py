#imports
import pandas as pd
import numpy as np

#read csv files and create dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train)

#comments to Andreia
#Name field might be relevant when NaN (convert to binary 'known_name') - higher probability of death or euthanasia (?) due to poorer condition when arriving at shelter
#0 probability of return_to_owner (? check no. NaN name observations per value of outcome)
#check dog/cat balance

#functions

def column_split(df,column,sep,n,new_col):
	'''function to create new columns from the original one when 2 or more attributes are stored in the same column, returns the new dataframe'''
	'''df is the original dataframe, column is the column to split, sep is the separator between attributes, n is the number of new columns, new_col is the list of new column names in the order that the attributes appear in the original column'''
	i=0
	while i < n:
		df[new_col[i]] = df[column].str.split(sep,n,True)[i]
		i+=1
	return df
	
def age_standard(row):
	'''function to convert the age to a standard numerical measurement in days (smallest unit of measurement present)'''
	if 'day' in str(row):
		val = float(row.split(' ')[0])
	elif 'week' in str(row):
		val = float(row.split(' ')[0])*7
	#it is assumed that a year has 365 days, so a month has 30.4 days
	elif 'month' in str(row):
		val = round(int(row.split(' ')[0])*30.4)
	elif 'year' in str(row):	
		val = round(int(row.split(' ')[0])*365)
	else:
		val = None
	return val

def pure_mix(row):
	'''function to create a column that flags if an animal is pure breed or mix'''
	if 'Mix' in str(row):
		val='mix'
	elif '/' in str(row):
		val='mix'
	else:
		val='pure'
	return val

def dangerous_breed(row):
	'''function to create a column that flags dangerous breeds'''
	dangerous_list = ['Staffordshire Bull Terrier', 'American Pit Bull Terrier', 'American Staffordshire Terrier', 'Dogo Argentino', 'American Bulldog', 'Tosa Inu', 'Dogo Canario', 'Cane Corso', 'Fila Brasileiro', 'Brazilian Mastiff', 'Akita Inu', 'Dogue De Bordeaux', 'Bandog', 'Bullmastiff', 'Doberman Pinsch', 'Ca de Bou', 'Majorca Mastiff','Kuvasz', 'German Shepherd', 'Mastino Neapolitano', 'Neapolitan Mastiff',  'Rottweiler', 'Chow Chow', 'Japanese Mastiff']
	if any(i in str(row) for i in dangerous_list):
		val = 'dangerous'
	else:
		val = 'ok'
	return val

	
#main	
df = column_split(train, 'SexuponOutcome', ' ', 2, ['sex_intervention', 'sex'])
#how to treat Unknown in sex_intervention (string) and None in sex (null value)?

#.apply allows to apply functions to each row or column
df['age'] = df['AgeuponOutcome'].apply(age_standard,1)
#0 years = 0.0 and blanks kept as nulls. Convert all to the same way of treating nulls? Which? Or what does 0 years mean?

df['pure_mix'] = df['Breed'].apply(pure_mix,1)

df['dangerous'] = df['Breed'].apply(dangerous_breed,1)

#for all attributes look at no. observations per attribute_value vs outcome_value 
#age
print(pd.pivot_table(df,values=['age'],columns = ['OutcomeType'], aggfunc = [min,max,np.mean]))
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min

#sex
print(pd.pivot_table(df,index=['sex','sex_intervention'], columns = ['OutcomeType'], aggfunc = 'count'))
#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine

#Breed
pd.pivot_table(df,index=['Breed'], columns = ['OutcomeType'], aggfunc = 'count').to_csv('breed_pivot.csv')
#what else can be done with breed?
#cats are less likely to be returned to owner ?
#some breeds have very few observations, to use this attribute a more summarised one will be needed (size, country of origin, life expectancy?)

#Colour
#Specified when 1 or 2, more than 2 as 'tricolored'; worth creating a no. colours variable? Doesn't seem useful
pd.pivot_table(df,index=['Color'], columns = ['OutcomeType'], aggfunc = 'count').to_csv('colour_pivot.csv')
#unbalanced attribute, with Black and Black/White having considerably more observations than other colours
#only use I can envision for colour would be for likelihood of adoption, with some colours being more likely to be adopted, but can't see any trend

#will need to repeat for test or can create a 'pre_processing' function that applies all final changes that we decide and then apply this to both train and test
#df.to_csv('dataframe.csv')

