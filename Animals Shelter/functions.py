#comments to Daniela
#similarity threshold in dangerous traits
#check possible values for SexUponOutcome

#functions

def column_split(df,column,sep,n,new_col):
	'''function to create new columns from the original one when 2 or more attributes are stored in the same column, returns the new dataframe'''
	'''df is the original dataframe, column is the column to split, sep is the separator between attributes, n is the number of new columns, new_col is the list of new column names in the order that the attributes appear in the original column'''
	i=0
	while i < n:
		df[new_col[i]] = df[column].str.split(sep,n,True)[i]
		i+=1
	return df