#functions
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

def pre_process(dataset):
		
	df=dataset.copy()  
	df=cat_coat(df)
	
	if 'OutcomeSubtype' in df.columns:
		df=df.drop(['OutcomeSubtype'], axis=1)
	
	df=process_date_attributes(df)
	df=process_name(df)
	df=process_sex(df)
	df=process_age(df)
	df=process_breed(df)
	df=process_colour(df)

	toDummy=["TimeOfDay", "AnimalType"]
	for column in toDummy:
		df = create_dummies(df, column)
	
	df=df.drop(['AnimalType','AnimalType_Cat','TimeOfDay_Morning', 'TimeOfDay'], axis=1)
	
	to_numeric = ['Winter', 'Christmas', 'Summer', 'Weekend']
	for column in to_numeric:
		df[column] = df[column].convert_objects(convert_numeric=True)
	
	return df

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
	df["Name"] = df["Name"].fillna("Unknown")
	
	#Missing names are labeled with Unknown and a new column is created 
	df["Unknown_Name"] = np.where((df["Name"]=="Unknown"),1,0)
	
	df['Common_Name_85'] = common_value(df['Name'],85)
	df['Common_Name_90'] = common_value(df['Name'],90)
	df['Common_Name_92'] = common_value(df['Name'],92)
	df['Common_Name_95'] = common_value(df['Name'],95)
	
	#Unknown names are removed from common names
	df["Common_Name_85"] = np.where((df['Name']== "Unknown"),0, df['Common_Name_85'])
	df["Common_Name_90"] = np.where((df['Name']== "Unknown"),0, df['Common_Name_90'])
	df["Common_Name_92"] = np.where((df['Name']== "Unknown"),0, df['Common_Name_92'])
	df["Common_Name_95"] = np.where((df['Name']== "Unknown"),0, df['Common_Name_95'])
	df=df.drop(['Name'], axis=1)
		
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

def known_age(row):
	'''function to convert nulls/unknown to unknown'''
	if row >= 0:
		val = 1
	else:
		val = 0
	return val

def age_bins(row):
	'''function to build age bins (based on days); apply only after replacing nulls by mean'''
	if row <= 30:
		val = '<= 1 month'
	elif row <= 61:
		val = '>1-2 months'
	elif row <= 243:
		val = '>2-8 months'
	elif row < 730:
		val = '9 months to 1 year'	
	elif row < 1095:
		val = '2 years'		
	elif row < 1825:
		val = '3-4 years'	
	elif row < 2920:
		val = '5-7 years'
	elif row < 4015:
		val = '8-10 years'			
	else:
		val = '>10 years'
	return val

def pure(row):
	'''function to create a column that flags if an animal is pure breed or mix'''
	if 'Mix' in str(row):
		val=0
	elif '/' in str(row):
		val=0
	else:
		val=1
	return val

def dangerous_breed(row):
	'''function to create a column that flags dangerous breeds'''
	dangerous_list = ['Staffordshire Bull Terrier', 'American Pit Bull Terrier', 'American Staffordshire Terrier', 'Dogo Argentino', 'American Bulldog', 'Tosa Inu', 'Dogo Canario', 'Cane Corso', 'Fila Brasileiro', 'Brazilian Mastiff', 'Akita Inu', 'Dogue De Bordeaux', 'Bandog', 'Bullmastiff', 'Doberman Pinsch', 'Ca de Bou', 'Majorca Mastiff','Kuvasz', 'German Shepherd', 'Mastino Neapolitano', 'Neapolitan Mastiff',  'Rottweiler', 'Chow Chow', 'Japanese Mastiff']
	if any(i in str(row) for i in dangerous_list):
		val = 1
	else:
		val = 0
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
	
def cat_coat(df):
	'''function to create a column with the cat coat'''
	#import cat coat information from excel file
	coat = pd.read_excel('cat_coat.xlsx')
	df = pd.merge(df, coat, how = 'left', on = 'Breed')
	return df

def breed_hypoaller(row):
	'''function to flag hypoallergenic breeds'''
	list = ['Balinese-Javanese','Cornish Rex','Devon Rex','Sphynx','Poodle','Yorkshire Terrier','Miniature Schnauzer','Shih Tzu','Havanese','Maltese','West Highland White Terrier',
	'Bichon Frise','Soft Coated Wheaten Terrier','Portuguese Water Dog','Airedale Terrier','Samoyed','Scottish Terrier','Wirehaired Pointing Griffon','Cairn Terrier','Italian Greyhound','Chinese Crested','Giant Schnauzer','Coton De Tulear','Bouvier des Flandres',
	'Standard Schnauzer', 'Border Terrier', 'Afghan Hound', 'Brussels Griffon', 'Wire Fox Terrier', 'Tibetan Terrier', 'Norwich Terrier', 'Silky Terrier', 'Welsh Terrier', 'Irish Terrier', 'Lagotto Romagnolo',
	'Norfolk Terrier', 'Kerry Blue Terrier', 'Australian Terrier', 'Lakeland Terrier', 'Puli', 'Xoloitzcuintli', 'Affenpinscher', 'Spanish Water Dog', 'Sealyham Terrier', 'Bedlington Terrier', 'Irish Water Spaniel',
	'Lowchen', 'Polish Lowland Sheepdog', 'Bergamasco', 'Dandie Dinmont Terrier', 'Cesky Terrier', 'Toy Poodle', 'Barbet', 'Peruvian Inca Orchid', 'Bolognese dog', 'Kyi-Leo', 'Shichon', 'Mountain Cur', 'Russian Tsvetnaya Bolonka']
	if any(i in str(row) for i in list):
		val = 1
	else:
		val = 0
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

def order_colour(row):
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
	df["Christmas"] = np.where((df["MonthTime"] == 12) & (df["DayTime"] >= 15), '1', (np.where((df["MonthTime"] == 1) & (df["DayTime"]<=10), '1', '0')))
	df["Summer"] = np.where((df["MonthTime"] >= 6) & (df["MonthTime"]<=9), '1', '0')
	df["Winter"] = np.where((df["MonthTime"] >= 12),1,(np.where((df["MonthTime"]<=2), '1', '0')))
	   
	df = transform_cyclic_attributes(df)
	df = extract_timeOfDay(df)
		
	#date_attributes = df[['DayTime', 'Month_sin','Month_cos','YearTime','Hour_sin', 'Hour_cos','WeekDay_sin','WeekDay_cos','Weekend', 'Christmas', 'Summer','Winter']]
	#date_attributes=date_attributes.reset_index()
	#df = pd.concat([df_original,date_attributes],axis=1)
	df=df.reset_index()
	df=df.drop(['Date','DateTime','DayTime', 'HourTime', 'WeekDay','MonthTime'], axis=1) #why not dropping other vars like sin and cos?
	return df

def process_sex(df):
	'''function to process the sex attribute'''
	df = column_split(df, 'SexuponOutcome', ' ', 2, ['sex_intervention', 'sex'])
	df['sex']= df['sex'].apply(unknown_value,1)
	df['sex_intervention']= df['sex_intervention'].apply(unknown_value,1).apply(sex_int,1)

	df = create_dummies(df,'sex')
	df = create_dummies(df,'sex_intervention')
	df=df.drop(['SexuponOutcome','sex','sex_intervention','sex_Unknown','sex_intervention_Unknown'], axis=1) #keep dummies only, right?
	return df
	
def process_age(df):
	'''function to process the age attribute'''
	df['age'] = df['AgeuponOutcome'].apply(age_standard,1)
	df['age_known']=df['age'].apply(known_age,1)
	df_trans = (df.groupby('AnimalType')).transform(lambda x: x.fillna(round(x.mean()), inplace = False)) 
	df['age'] = df_trans['age']
	df['age_bins']= df['age'].apply(age_bins,1)
	df = create_dummies(df,'age_bins')
	df= df.drop(['AgeuponOutcome','age_bins','age_bins_>10 years','age_known'], axis=1) 
	
	return df
	
def process_breed(df):
	'''function to process the breed attribute'''
	df['pure'] = df['Breed'].apply(pure,1)
	df['dangerous'] = df['Breed'].apply(dangerous_breed,1)
	df['size'] = df['Breed'].apply(breed_size,1)
	df['intelligence'] = df['Breed'].apply(breed_intelligence,1)
	df['hypoaller'] = df['Breed'].apply(breed_hypoaller,1)
	df['Coat'] = df['Coat'].apply(unknown_value,1)
	df = create_dummies(df,'size')
	df = create_dummies(df,'intelligence')
	df = create_dummies(df,'Coat')
	df=df.drop(['Breed','size','intelligence','size_other','intelligence_other','Coat','Coat_Unknown'], axis=1) 
	return df
	
def process_colour(df):
	'''function to process the colour attribute'''
	df['Colour'] = df['Color'].apply(order_colour,1)
	df['no_colours'] = df['Colour'].apply(breed_colour,1)
	df = create_dummies(df,'no_colours')
	df['common_colour_90'] = common_value(df['Colour'],90)
	df['common_colour_92'] = common_value(df['Colour'],92)
	df['common_colour_95'] = common_value(df['Colour'],95) #considers 16 most common colours as common (more than 550 occurrences)
	df['common_colour_98'] = common_value(df['Colour'],98)
	df=df.drop(['Color', 'Colour','no_colours','no_colours_unicolour'], axis=1) 

	return df
	
def feature_selection(df,tgt,mtd):
	'''function to do feature selection for the target specified by tgt and using the method specified by mtd'''
	target = df[tgt]
	features = df.drop([tgt], axis=1)
	if mtd == 'KBest':
		bestfeatures = SelectKBest(score_func=f_classif, k=10)
		fit = bestfeatures.fit(features,target)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(features.columns)
		#dfpvalues = pd.DataFrame(fit.pvalues_) #not providing useful insight
		featureScores = pd.concat([dfcolumns,dfscores],axis=1)
		featureScores.columns = ['Features','Score']  #naming the dataframe columns
		featureScores = featureScores.sort_values(by=['Score'],ascending=False) #sort to see most important features at the top
		select_cols = features.columns.values[fit.get_support()] #get_support returns a boolean vector indicating which features (columns' names) were selected
		selectfeatures = pd.DataFrame(fit.transform(features),columns = select_cols) #build dataframe with selected features only
	elif mtd == 'Fdr':
		bestfeatures = SelectFdr(score_func=f_classif, alpha=0.05)
		fit = bestfeatures.fit(features,target)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(features.columns)
		#dfpvalues = pd.DataFrame(fit.pvalues_) #not providing useful insight
		featureScores = pd.concat([dfcolumns,dfscores],axis=1)
		featureScores.columns = ['Features','Score']  #naming the dataframe columns
		featureScores = featureScores.sort_values(by=['Score'],ascending=False) #sort to see most important features at the top
		select_cols = features.columns.values[fit.get_support()] #get_support returns a boolean vector indicating which features (columns' names) were selected
		selectfeatures = pd.DataFrame(fit.transform(features),columns = select_cols) #build dataframe with selected features only
	elif mtd == 'Fwe':
		bestfeatures = SelectFwe(score_func=f_classif, alpha=0.05)
		fit = bestfeatures.fit(features,target)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(features.columns)
		#dfpvalues = pd.DataFrame(fit.pvalues_) #not providing useful insight
		featureScores = pd.concat([dfcolumns,dfscores],axis=1)
		featureScores.columns = ['Features','Score']  #naming the dataframe columns
		featureScores = featureScores.sort_values(by=['Score'],ascending=False) #sort to see most important features at the top
		select_cols = features.columns.values[fit.get_support()] #get_support returns a boolean vector indicating which features (columns' names) were selected
		selectfeatures = pd.DataFrame(fit.transform(features),columns = select_cols) #build dataframe with selected features only
	elif mtd == 'Pct':
		bestfeatures = SelectPercentile(score_func=f_classif, percentile=20)
		fit = bestfeatures.fit(features,target)
		dfscores = pd.DataFrame(fit.scores_)
		dfcolumns = pd.DataFrame(features.columns)
		#dfpvalues = pd.DataFrame(fit.pvalues_) #not providing useful insight
		featureScores = pd.concat([dfcolumns,dfscores],axis=1)
		featureScores.columns = ['Features','Score']  #naming the dataframe columns
		featureScores = featureScores.sort_values(by=['Score'],ascending=False) #sort to see most important features at the top
		select_cols = features.columns.values[fit.get_support()] #get_support returns a boolean vector indicating which features (columns' names) were selected
		selectfeatures = pd.DataFrame(fit.transform(features),columns = select_cols) #build dataframe with selected features only

	return featureScores #selectfeatures

def convert(data, to):
	converted = None
	if to == 'array':
		if isinstance(data, np.ndarray):
			converted = data
		elif isinstance(data, pd.Series):
			converted = data.values
		elif isinstance(data, list):
			converted = np.array(data)
		elif isinstance(data, pd.DataFrame):
			converted = data.as_matrix()
	elif to == 'list':
		if isinstance(data, list):
			converted = data
		elif isinstance(data, pd.Series):
			converted = data.values.tolist()
		elif isinstance(data, np.ndarray):
			converted = data.tolist()
	elif to == 'dataframe':
		if isinstance(data, pd.DataFrame):
			converted = data
		elif isinstance(data, np.ndarray):
			converted = pd.DataFrame(data)
	else:
		raise ValueError("Unknown data conversion: {}".format(to))
	if converted is None:
		raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
	else:
		return converted
		
def correlation_ratio(categories, measurements):
	'''	:param categories: list / NumPy ndarray / Pandas Series
	A sequence of categorical measurements
	:param measurements: list / NumPy ndarray / Pandas Series
	A sequence of continuous measurements
	:return: float
	in the range of [0,1]'''
	cat = convert(categories, 'array')
	measurements = convert(measurements, 'array')
	fcat, _ = pd.factorize(cat)
	cat_num = np.max(fcat)+1
	y_avg_array = np.zeros(cat_num)
	n_array = np.zeros(cat_num)
	for i in range(0,cat_num):
		cat_measures = measurements[np.argwhere(fcat == i).flatten()]
		n_array[i] = len(cat_measures)
		y_avg_array[i] = np.average(cat_measures)
	y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
	numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
	denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
	if numerator == 0:
		eta = 0.0
	else:
		eta = numerator/denominator
	return eta