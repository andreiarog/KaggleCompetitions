#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions as func #file containing the user created functions

#read csv files and create dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
	
#main
#.apply allows to apply functions to each row or column


df=func.pre_process(train)
df.to_csv('dataframe.csv') 

features_select_K = func.feature_selection(df.drop(['AnimalID'],axis=1),'OutcomeType','KBest') #need to concat with class to obtain final dataframe for classification
#features_select_Fdr = func.feature_selection(df.drop(['AnimalID'],axis=1),'OutcomeType','Fdr') #need to concat with class to obtain final dataframe for classification
#features_select_Fwe = func.feature_selection(df.drop(['AnimalID'],axis=1),'OutcomeType','Fwe') #need to concat with class to obtain final dataframe for classification
#features_select_Pct = func.feature_selection(df.drop(['AnimalID'],axis=1),'OutcomeType','Pct') #need to concat with class to obtain final dataframe for classification

print('KBest:',features_select_K)
#print('Fdr:',features_select_Fdr)
#print('Fwe:',features_select_Fwe)
#print('Fwe:',features_select_Pct)
#print('KBest:',features_select_K.columns)
#print('Pct:',features_select_Pct.columns)

#correlation between features, which are all numerical
feature_corr = df.drop(['AnimalID'],axis=1).corr()
#print(feature_corr)

#correlation between features and class using the correlation ratio
corr_r = []
for i in df.drop(['AnimalID','OutcomeType'],axis=1).columns: #for each category, calculate the correlation ratio with the class. Function from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
	corr_r.append(func.correlation_ratio(df['OutcomeType'],df[i]))

dfratios = pd.DataFrame(corr_r,columns = ['OutcomeType'], index=feature_corr.columns) #need to define same indexes as feature_corr to properly concatenate dataframes
#print(dfratios)
corr_r.append(1.000000)
columns = ['YearTime','Weekend','Christmas','Summer','Winter','Hour_sin','Hour_cos','Month_sin','Month_cos','WeekDay_sin','WeekDay_cos','Unknown_Name','Common_Name_85','sex_Female','sex_Male','sex_intervention_Intact','sex_intervention_Intervention','age','age_bins_2 years','age_bins_3-4 years','age_bins_5-7 years','age_bins_8-10 years','age_bins_9 months to 1 year','age_bins_<= 1 month','age_bins_>1-2 months','age_bins_>2-8 months','pure','dangerous','hypoaller','size_giant','size_large','size_medium','size_small','size_toy','intelligence_above average','intelligence_average','intelligence_bright','intelligence_excellent','intelligence_fair','intelligence_lowest','no_colours_bicolour','no_colours_multicolour','common_colour_98','TimeOfDay_Afternoon','TimeOfDay_Evening','AnimalType_Dog','OutcomeType']

ratios = dict(zip(columns, corr_r)) #create dictionary to transform in 1 row dataframe
rowratios = pd.DataFrame(ratios,index=['OutcomeType']) #since dictionary is in scalar form and not list need to pass an index, which will also be needed for the heatmap (see feature_corr structure)

#plot heat map
correlations = pd.concat([feature_corr,dfratios],axis=1) #add column with correlation with OutcomeType
correlations = correlations.append(rowratios) #add row with correlation with OutcomeType, need to append 1 as the correlation of the class with itself
#print(correlations)

plt.figure(figsize=(20,20))
g=sns.heatmap(correlations,cmap="RdYlGn")
plt.show(g)

print('Feature importance:' , func.important_features_PCA (df.drop(['AnimalID','OutcomeType'],axis=1)))

#feature selection comments:
#hour sin and cos have good ranking, how to use them?
#all methods provide the same scores: all features are correlated with outcome, but some more strongly, according to scores' order
#correlations in heatmap are aligned with feature selection results: with unknown name and intervention variables being the most strongly correlated with the outcome
#we need to select non-repeated features according to the scores
#how many features should we select? Just try the methods with different number of features and select best? 
#think we can check feature importance and impact of using different features when testing classifiers, but maybe not as a pre-step given that the choice will be biased by the specific method being used and is only possible for tree based classifiers
#Top features from correlation are also high in PCA importance rank
#final features: keep common_name_85 and common_colour_98; keep age and bins but never use together (same for weekend and weekday_sin)
#classifiers: linear SVC, random forest, logistic regression with multinomial distribution

#OAO baseline classification
func.OAO_classif(df.drop(['AnimalID'],axis=1), 'OutcomeType')

#baseline classification: compare onevsone, onevsall and all&one approaches with each other and Andreia's baseline (starting with all variables and imbalanced sample); use not only accuracy but also confusion matrix and multiclass AUC
#no proved best approach for imbalanced multiclass problems, depends on the dataset

#dataset unbalanced towards neutered/spayed, which is more than double 'intact' animals (might need to omit this variable); gender balance is fine
#good correlation between sex_intervention and outcome	
#there are no adoptions of animals younger than a month
#the average age for death is the lowest whereas for euthanasia is the 2nd highest, it seems that death is correlated with younger animals unable to survive arsh conditions
#adoption and transfer have a lower mean, with transfer and return to owner having the biggest max-min
#can't find a good size and intelligence list for cats; breed info will be very different based on animal type; should classify separately?
#size and intelligence list for dogs still incomplete, but covering most frequent breeds (better for size)


#unbalanced attribute, with Black and Black/White having considerably more observations than other colours




