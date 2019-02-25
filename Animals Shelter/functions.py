#functions

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
	
def unknown_value(row):
	'''function to convert nulls/unknown to unknown'''
	str1 = 'unknown'
	str2 = ' '
	if str(row).lower()== str1.lower() or row == None or str(row) == 'nan':
		val = 'Unknwon'
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
	

