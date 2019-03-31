#functions
import pandas as pd
import numpy as np

def column_split(df,column,sep,n,new_col):
	'''function to create new columns from the original one when 2 or more attributes are stored in the same column, returns the new dataframe'''
	'''df is the original dataframe, column is the column to split, sep is the separator between attributes, n is the number of new columns, new_col is the list of new column names in the order that the attributes appear in the original column'''
	i=0
	while i < n:
		df[new_col[i]] = df[column].str.split(sep,n,True)[i]
		i+=1
	return df

def create_dummies(df,column_name):
	"""Create Dummy Columns (One Hot Encoding) from a single Column

	Usage
	------
	train = create_dummies(train,"Age")
	"""
	dummies = pd.get_dummies(df[column_name],prefix=column_name)
	df = pd.concat([df,dummies],axis=1)
	return df
	
def normalize(df,column_name, mode):
	"""Normalize Continuous Columns 

	Usage
	------
	train = create_dummies(train,"Age")
	"""
	if (mode=="n"):
		column=df[column_name]
		new="{}_norm".format(column_name)
		df[new]=(column-column.min())/(column.max()-column.min())
		df[column_name]=df[new]
		df=df.drop([new], axis=1)
	if (mode=="z"):
		column=df[column_name]
		new="{}_norm".format(column_name)
		avg=column.mean()
		var=df.loc[:,column_name].var()
		df[new]=(column-avg)/var
		df[column_name]=df[new]
		df=df.drop([new], axis=1)
		
	
	return df
	
def common_value(col,pct):
	'''function to identify common and uncommon values of a column'''
	
	#The frequency of each name is stored and a minimum counting is defined as 90 percentile of names sorted (it's 6 times)
	countings = col.value_counts()	
	countings_values = col.value_counts().tolist()
	minimum_counting = np.percentile(countings_values, pct, axis=0)
	
	#A new column is created with names having a frequency higher than 90 percentile defined previously
	new_col = np.where((countings[col]>= minimum_counting),1, 0)
	
	return new_col

def process_name(dataset): #needs to be tested

	df = dataset.copy()
	
	#Missing names are labeled with Unknown and a new column is created 
	df["Name"] = df["Name"].fillna("Unknown")
	
	df["Unknown_Name"] = np.where((df["Name"]=="Unknown"),1,0)
	
	df['Common_Name'] = common_value(df['Name'],90)
	
	#Unknown names are removed from common names
	df["Common_Name"] = np.where((df['Name']== "Unknown"),0, df['Common_Name'])
	
	return df

def sex_int(row):
	'''function to rename as Intervention all animals that were either neutered or spayed'''
	if 'Unknown' in str(row):
		val='Unknown'
	elif 'Intact' in str(row):
		val='Intact'
	else:
		val='Intervention'
	return val
	
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

def unknown_age(row):
	'''function to convert nulls/unknown to unknown'''
	if row >= 0:
		val = 'Known'
	else:
		val = 'Unknown'
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
	
def breed_size(row):
	'''function to create a column with the breed's size'''
	giant_list = ['Akita', 'Anatolian Sheepdog','Anatol Shepherd','Bernese Mountain Dog','Bloodhound','Borzoi','Bullmastiff','Great Dane','Great Pyrenees',
'Great Swiss Mountain Dog','Irish Wolfhound','Kuvasz','Mastiff','Neopolitan Mastiff','Newfoundland','Otterhound','Rottweiler','Saint Bernard']
	large_list = ['Afghan Hound','Alaskan Malamute','American Foxhound','American Bulldog','Beauceron','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren',
'Black And Tan Coonhound','Black Russian Terrier','Bouvier Des Flandres','Boxer','Briard','Chesapeake Bay Retriever','Clumber Spaniel','Collie',
'Curly Coated Retriever','Doberman Pinsch','English Foxhound','English Setter','Flat Coat Retriever','German Shepherd','German Shorthaired Pointer',
'German Wirehaired Pointer','Giant Schnauzer','Golden Retriever','Gordon Setter','Greyhound','Irish Setter','Komondor','Labrador Retriever',
'Old English Sheepdog','Poodle Standard','Rhodesian Ridgeback','Rhod Ridgeback','Scottish Deerhound','Spinone Italiano','Tibetan Mastiff','Weimaraner']
	medium_list = ['Airedale Terrier','American Staffordshire Terrier','American Water Spaniel','Alaskan Husky','Australian Cattle Dog','Australian Shepherd','Basset Hound','Bearded Collie','Black Mouth Cur','Blue Lacy','Border Collie',
'Brittany','Bulldog','Bull Terrier','Canaan Dog','Carolina Dog','Catahoula','Chinese Shar Pei','Chow Chow','Cocker Spaniel',
'Dalmatian','English Springer Spaniel','Field Spaniel','Flat Coated Retriever','Finnish Spitz','Harrier','Ibizan Hound','Irish Terrier',
'Irish Water Spaniel','Keeshond','Kerry Blue Terrier','Norwegian Elkhound','Nova Scotia Duck Tolling Retriever','Petit Basset Griffon Vendeen',
'Pharaoh Hound','Pit Bull','Plott Hound','Pointer','Polish Lowland Sheepdog','Portuguese Water Dog','Queensland Heeler','Redbone Coonhound','Redbone Hound','Saluki','Samoyed','Siberian Husky',
'Soft Coated Wheaten Terrier','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Vizsla','Welsh Springer Spaniel','Wirehaired Pointing Griffon']
	small_list = ['American Eskimo','Australian Terrier','Australian Kelpie','Basenji','Beagle',
'Bedlington Terrier','Bichon Frise','Border Terrier','Boston Terrier','Brussels Griffon','Bruss Griffon','Cairn Terrier','Cardigan Welsh Corgi',
'Cavalier King Charles Spaniel','Coton de Tulear','Dachshund','Dandie Dinmont Terrier','English Toy Spaniel','Fox Terrier – Smooth',
'Wire Hair Fox Terrier','French Bulldog','German Pinscher','Glen Imaal Terrier','Jack Russell Terrier','Lhasa Apso','Lakeland Terrier','Manchester Terrier (Standard)',
'Manchester Terrier','Miniature',' Miniature Poodle','Norfolk Terrier','Norwich Terrier','Pug','Puli','Pembroke Welsh Corgi','Rat Terrier','Schipperke','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu',
'Shih Tzu','Silky Terrier','Skye Terrier','Staffordshire','Tibetan Spaniel','Tibetan Terrier','Welsh Terrier','West Highland','Whippet']
	toy_list = ['Affenpinscher','Chihuahua','Chinese Crested','Italian Greyhound','Japanese Chin','Maltese','Manchester Terrier (Toy)','Papillon','Pekingese','Pomeranian',' Toy Poodle','Toy Fox Terrier','Yorkshire Terrier']
	if any(i in str(row) for i in giant_list):
		val = 'giant'
	elif any(i in str(row) for i in large_list):
		val = 'large'
	elif any(i in str(row) for i in medium_list):
		val = 'medium'
	elif any(i in str(row) for i in small_list):
		val = 'small'
	elif any(i in str(row) for i in toy_list):
		val = 'toy'
	else:
		val = 'other'
	return val

def breed_intelligence(row):
	'''function to create a column with the breed's intelligence'''
	bright_list = ['Border Collie','Poodle','German Shepherd Dog','Golden Retriever','Doberman Pinscher','Shetland Sheepdog','Labrador Retriever','Papillon','Rottweiler','Australian Cattle Dog','German Shepherd']
	excellent_list = ['Catahoula','Black Mouth Cur','Pembroke Welsh Corgi','Miniature Schnauzer','English Springer Spaniel','Belgian Shepherd Dog (Tervuren)','Pit Bull',
'Schipperke','Belgian Sheepdog','Collie','Keeshond','German Shorthaired Pointer','Flat Coat Retriever','English Cocker Spaniel','Standard Schnauzer','Brittany','Cocker Spaniel',
'Weimaraner','Belgian Malinois','Bernese Mountain Dog','Pomeranian','Irish Water Spaniel','Vizsla','Cardigan Welsh Corgi']
	above_avg_list = ['Doberman Pinsch','Anatol Shepherd','Australian Kelpie','Chesapeake Bay Retriever','Puli','Yorkshire Terrier','Giant Schnauzer','Portuguese Water Dog','Airedale Terrier','Bouvier des Flandres','Border Terrier',
'Briard','Welsh Springer Spaniel','Australian Shepherd','Manchester Terrier','Samoyed','Field Spaniel','Newfoundland','Australian Terrier','American Staffordshire Terrier',
'Gordon Setter','Bearded Collie','Cairn Terrier','Kerry Blue Terrier','Irish Setter','Norwegian Elkhound','Affenpinscher',
'Australian Silky Terrier','Miniature Pinscher','English Setter','Pharaoh Hound','Clumber Spaniel','Norwich Terrier','Dalmatian']
	avg_list = ['Rat Terrier','Blue Lacy','Soft Coated Wheaten Terrier','Bedlington Terrier','Smooth Fox Terrier','Curly Coated Retriever','Irish Wolfhound','Kuvasz','Australian Shepherd','Saluki','Finnish Spitz','Pointer',
'Cavalier King Charles Spaniel','German Wirehaired Pointer','Black/Tan Hound','Black and Tan Coonhound','American Water Spaniel','Siberian Husky','Bichon Frise','Havanese','King Charles Spaniel','Tibetan Spaniel','English Foxhound',
'Otterhound','Jack Russell Terrier','American Foxhound','Greyhound','Wirehaired Pointing Griffon','West Highland White Terrier','Scottish Deerhound','Boxer','Great Dane','Dachshund','Shiba Inu',
'Staffordshire Bull Terrier','Alaskan Malamute','Whippet','Chinese Shar Pei','Wire Fox Terrier','Rhodesian Ridgeback','Ibizan Hound','Welsh Terrier','Irish Terrier','Boston Terrier','Akita']
	fair_list = ['Skye Terrier','Norfolk Terrier','Sealyham Terrier','Pug','French Bulldog','Griffon Bruxellois','Maltese','Italian Greyhound','Coton de Tulear','Chinese Crested','Dandie Dinmont Terrier',
'Petit Basset Griffon Vendéen','Tibetan Terrier','Japanese Chin','Lakeland Terrier','Old English Sheepdog','Great Pyrenees','Scottish Terrier','Saint Bernard','Bull Terrier','Chihuahua','Lhasa Apso','Bullmastiff']
	lowest_list = ['Shih Tzu','Basset Hound','Mastiff','Beagle','Pekingese','Bloodhound','Borzoi','Chow Chow','Bulldog','Basenji','Afghan Hound']
	if any(i in str(row) for i in bright_list):
		val = 'bright'
	elif any(i in str(row) for i in excellent_list):
		val = 'excellent'
	elif any(i in str(row) for i in above_avg_list):
		val = 'above average'
	elif any(i in str(row) for i in avg_list):
		val = 'average'
	elif any(i in str(row) for i in fair_list):
		val = 'fair'
	elif any(i in str(row) for i in lowest_list):
		val = 'lowest'
	else:
		val = 'other'
	return val
	
def breed_colour(row):
	'''function to create a column that flags unicolour vs multi-colour'''
	if 'Tricolor' in str(row):
		val = 'multicolour'
	elif '/' in str(row):
		val = 'bicolour'
	else:
		val = 'unicolour'
	return val

def process_colour(row):
	'''function to process colour names'''
	if '/' in str(row):
		colours = str(row).split('/')
		if colours[0] == colours[1]:
			val=colours[0]
		elif colours[0] > colours[1]:
			val=colours[0]+'/'+colours[1]
		else:
			val=colours[1]+'/'+colours[0]
	else:
		val = str(row)
	return val
	
def unknown_value(row):
	'''function to convert nulls/unknown to unknown'''
	str1 = 'unknown'
	str2 = ' '
	if str(row).lower()== str1.lower() or row == None or str(row) == 'nan':
		val = 'Unknown'
	else:
		val = str(row)
	return val

#date functions	
def extract_timeOfDay(df):
	conditions = [
	(df['HourTime'] >= 6) & (df['HourTime'] <= 11),
	(df['HourTime'] >= 12) & (df['HourTime'] <= 18),
	(df['HourTime'] >= 19),
	(df['HourTime'] <= 5)]
	
	choices = ['Morning', 'Afternoon', 'Evening', 'Evening']
	
	df['TimeOfDay'] = np.select(conditions, choices, default='UnknownTimeOfDay')
	
	return df

def transform_cyclic_attributes(df):
	df['Hour_sin'] = np.sin(df['HourTime']*(2.*np.pi/24))
	df['Hour_cos'] = np.cos(df['HourTime']*(2.*np.pi/24))
	df['Month_sin'] = np.sin((df['MonthTime']-1)*(2.*np.pi/12))  #remove 1 to start at month 0
	df['Month_cos'] = np.cos((df['MonthTime']-1)*(2.*np.pi/12))
	df['WeekDay_sin'] = np.sin((df['WeekDay'])*(2.*np.pi/7))
	df['WeekDay_cos'] = np.cos((df['WeekDay'])*(2.*np.pi/7))
	
	return df
	
def extract_date_attributes(df):
	df['Date'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', utc=True)
	df.set_index('Date', inplace=True)
	df.index = df.index.round('H')
	
	df["DayTime"] = df.index.day   #not important as feature, just as condition of Christmas time
	df["MonthTime"] = df.index.month
	df["YearTime"] = df.index.year
	df["HourTime"] = df.index.hour
	df["WeekDay"] = df.index.weekday #Monday=0, Sunday=6
	df['Weekend'] = np.where((df["WeekDay"] >= 5), '1', '0')
	df["Christmas"] = np.where((df["MonthTime"] == 12) & (df["DayTime"] >= 15), '1', (np.where(df["MonthTime"] == 1, '1', '0')))
	df["Summer"] = np.where((df["MonthTime"] >= 6) & (df["MonthTime"]<=9), '1', '0')
	df["Winter"] = np.where((df["MonthTime"] >= 11) & (df["MonthTime"]<=2), '1', '0')
   
	df = transform_cyclic_attributes(df)
	df = extract_timeOfDay(df)
	
	return df
	
def process_date_attributes(dataset):
	df_original = dataset.copy()
	df=dataset.copy()
		
	df['Date'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', utc=True)
	df.set_index('Date', inplace=True)
	df.index = df.index.round('H')
		
	df["DayTime"] = df.index.day   #not important as feature, just as condition of Christmas time
	df["MonthTime"] = df.index.month
	df["YearTime"] = df.index.year
	df["HourTime"] = df.index.hour
	df["WeekDay"] = df.index.weekday #Monday=0, Sunday=6
	df['Weekend'] = np.where((df["WeekDay"] >= 5), '1', '0')
	df["Christmas"] = np.where((df["MonthTime"] == 12) & (df["DayTime"] >= 15), '1', (np.where(df["MonthTime"] == 1, '1', '0')))
	df["Summer"] = np.where((df["MonthTime"] >= 6) & (df["MonthTime"]<=9), '1', '0')
	df["Winter"] = np.where((df["MonthTime"] >= 11) & (df["MonthTime"]<=2), '1', '0')
	   
	df = transform_cyclic_attributes(df)
	df = extract_timeOfDay(df)
		
	#date_attributes = df[['DayTime', 'Month_sin','Month_cos','YearTime','Hour_sin', 'Hour_cos','WeekDay_sin','WeekDay_cos','Weekend', 'Christmas', 'Summer','Winter']]
	#date_attributes=date_attributes.reset_index()
	#df = pd.concat([df_original,date_attributes],axis=1)
	df=df.reset_index()
	df=df.drop(['Date','DateTime'], axis=1)
	return df

