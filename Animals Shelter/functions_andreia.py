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

def extract_timeOfDay(df):
    conditions = [
    (df['HourTime'] >= 6) & (df['HourTime'] <= 11),
    (df['HourTime'] >= 12) & (df['HourTime'] <= 18),
    (df['HourTime'] >= 19),
    (df['HourTime'] <= 5)]
    
    choices = ['Morning', 'Afternoon', 'Evening', 'Evening']
    
    df['TimeOfDay'] = np.select(conditions, choices, default='UnknownTimeOfDay')
    df = df.drop('HourTime',axis=1)
    return df

def transform_cyclic_attributes(df):
    df['Hour_sin'] = np.sin(df['HourTime']*(2.*np.pi/24))
    df['Hour_cos'] = np.cos(df['HourTime']*(2.*np.pi/24))
    df['Month_sin'] = np.sin((df['MonthTime']-1)*(2.*np.pi/12))  #remove 1 to start at month 0
    df['Month_cos'] = np.cos((df['MonthTime']-1)*(2.*np.pi/12))
    df['WeekDay_sin'] = np.sin((df['WeekDay'])*(2.*np.pi/7))
    df['WeekDay_cos'] = np.cos((df['WeekDay'])*(2.*np.pi/7))
    df=df.drop(df[['MonthTime','WeekDay']], axis=1)
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

def process_name(dataset):
    df = dataset.copy()
    
    #Missing names are labeled with Unknown and a new column is created 
    df["Name"] = df["Name"].fillna("Unknown")
    df["Unknown_Name"] = np.where((df["Name"]=="Unknown"),1,0)
    
    #The frequency of each name is stored and a minimum counting is defined as 90 percentile of names sorted (it's 6 times)
    countings = df["Name"].value_counts()    
    countings_values = df["Name"].value_counts().tolist()
    minimum_counting = np.percentile(countings_values, 90, axis=0)
    
    #A new column is created with names having a frequency higher than 90 percentile defined previously
    df["Common_Name"] = np.where((countings[df['Name']]>= minimum_counting),1, 0)
    #Unknown names are removed from common names
    df["Common_Name"] = np.where((df['Name']== "Unknown"),0, df['Common_Name'])
    
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------
    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop(column_name, axis=1)
    
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