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

#functions

def column_split(df,column,sep,n,new_col):
	'''function to create new columns from the original one when 2 or more attributes are stored in the same column, returns the new dataframe'''
	'''df is the original dataframe, column is the column to split, sep is the separator between attributes, n is the number of new columns, new_col is the list of new column names in the order that the attributes appear in the original column'''
	df[new_col[0]] = df[column].str.split(sep,n,True)[0]
	df[new_col[1]] = df[column].str.split(sep,n,True)[1]
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

df['age'] = df['AgeuponOutcome'].apply(age_standard,1)
#0 years = 0.0 and blanks kept as nulls. Convert all to the same way of treating nulls? Which? Or what does 0 years mean?

df['pure_mix'] = df['Breed'].apply(pure_mix,1)

df['dangerous'] = df['Breed'].apply(dangerous_breed,1)
#what else can be done with breed?

#colour - specified when 1 or 2, more than 2 as 'tricolored'; worth creating a no. colours variable? Doesn't seem useful

#for all attributes look at no. observations per attribute_value x outcome_value to see if other derived columns might be useful

#will need to repeat for test or can create a 'pre_processing' function that applies all final changes that we decide and then apply this to both train and test
#df.to_csv('dataframe.csv')

