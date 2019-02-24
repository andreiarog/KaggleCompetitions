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
