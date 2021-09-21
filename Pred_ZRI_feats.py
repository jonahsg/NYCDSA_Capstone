import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV

from Feat_eng import *

# zori - need to pull target DF from other module
zri_yr = target[['RegionName', 'year', 'value']].copy()

# loading zori
zori = pd.read_csv('../data/Zip_ZORI_AllHomesPlusMultifamily_SSA.csv')

# converting zip column (RegionName) to string and filling with 0s
zori['RegionName'] = zori['RegionName'].astype(str).str.zfill(5)

# flattening monthly rent indices
cols_replace = list(zori.columns)[4:]
cols_keep = list(zori.columns)[:4]
zori_new = zori.melt(id_vars=cols_keep, value_vars = cols_replace)

# converting month column to datetime object and pulling year
zori_new['variable']= pd.to_datetime(zori_new['variable'])
zori_new['year'] = zori_new['variable'].apply(lambda a: a.year)

# grouping by year column and calculating mean
zori_yr = zori_new.groupby(['RegionName', 'year']).mean().reset_index()

# dropping non used columns
zori_yr.drop(['SizeRank','RegionID'], inplace=True, axis=1)

df_comb3 = zri_yr.merge(zori_yr, how='left',
              left_on = ['RegionName', 'year'], 
              right_on = ['RegionName', 'year'])
df_comb3.columns = ['RegionName', 'year', 'zri', 'zori']

df_comb3_red = df_comb3.dropna()

# log transformation of both ZRI and ZORI values
df_comb3_red['log_zri'] = np.log(df_comb3_red['zri'])
df_comb3_red['log_zori'] = np.log(df_comb3_red['zori'])

# standardizing year values (simple subtraction of 2013 to keep values between 1 and 10)
df_comb3_red['year'] = df_comb3_red['year'] - 2013

# creating table grouped by year to find mean zri / zori values
test3 = df_comb3_red.copy()
test3[['zri_pred']] = 0

# using annual ZORI rates to project yearly ZRI indices
test3['zri_pred'][test3['year']==1] = (test3['zri']*0.049357)+test3['zri']
test3['zri_pred'][test3['year']==2] = (test3['zri']*0.036527)+test3['zri']
test3['zri_pred'][test3['year']==3] = (test3['zri']*0.028361)+test3['zri']
test3['zri_pred'][test3['year']==4] = (test3['zri']*0.029733)+test3['zri']
test3['zri_pred'][test3['year']==5] = np.nan

test3 = test3[test3['year']<5]

df_comb5 = df_comb3_red.copy()
df_comb5['year'] = df_comb5['year']-1

# merging ZRI predictions with corresponding ZRI values (correct year)
# renaming columns

test2 = test3.merge(df_comb5[['RegionName', 'zri', 'year']], how = 'left', left_on = ['RegionName', 'year'], right_on = ['RegionName','year'])
test2.columns = [['RegionName', 'year', 'zri', 'zori', 'log_zri', 'log_zori', 'zri_pred', 'zri_future']]

# dropping 6 rows w/ null values in zri_future
test2.dropna(inplace=True)

zori_tot = zori_yr.copy()

zips_unique_that_are_also_in_target = list(target['RegionName'].unique())

zori_new = zori_tot[zori_tot['RegionName'].isin(zips_unique_that_are_also_in_target)]

# mean zori values by year
zori_new = zori_new.groupby('year').mean('value').reset_index()

# percent change of zori values on an annual basis
zori_new['zori_change'] = zori_new['value'].pct_change()

# initializing DF with ZRI predictions in 2019, 2020, and 2021
testing = pd.DataFrame(np.repeat(zips_unique_that_are_also_in_target,3,axis=0))
testing.columns = ['RegionName']

# filling year values per zip code--> 2019 - 2021
from itertools import cycle
seq = cycle([2019, 2020, 2021])
testing['year'] = [next(seq) for count in range(testing.shape[0])]
testing

# filtering ZRI values for 2019 based off 2018 future values
zri_2019 = target[target['year']==2018][['RegionName', 'future_value']]
zri_2019['year'] = 2019

# merging 2019 ZRI values to testing
testing = testing.merge(zri_2019, how='left', on=['RegionName', 'year'])
testing.columns = ['RegionName', 'year', 'zri']

# calculating 2020 ZRI values using 2019-2020 ZORI percent change
zri_2020 = zri_2019[['RegionName']]
zri_2020['zri'] = (zri_2019[['future_value']]*0.003800)+zri_2019[['future_value']]
zri_2020['year'] = 2020

# merging 2020 ZRI predictions to testing (ZRI predictions) DF
testing = testing.merge(zri_2020, how='left', on=['RegionName', 'year'])

# calculating 2021 ZRI values using 2020-2021 ZORI percent change
zri_2021 = zri_2020[['RegionName']]
zri_2021['zri'] = (zri_2020[['zri']]*0.024881)+zri_2020[['zri']]
zri_2021['year'] = 2021

# merging 2021 ZRI predictions to testing (ZRI predictions) DF
testing = testing.merge(zri_2021, how='left', on=['RegionName', 'year'])

# setting null values to 0 and adding ZRI predictions across the rows
# dropping incomplete ZRI columns after summing
testing = testing.fillna(value=0)
testing['ZRI'] = testing['zri_x']+testing['zri_y']+testing['zri']
testing.drop(['zri_x', 'zri_y', 'zri'], axis=1, inplace=True)

# renaming final 2019 - 2021 predictions DF to future_target
future_target = testing.copy()

# creating working copy of target
new_target = target.copy()

# dropping columns not needed
new_target.drop(['geo_id', 'City', 'zip', 'future_value', 'value'], axis=1, inplace=True)

# checking to ensure columns of new_target and training set matches (besides RegionName)
main_list = list(set(new_target.columns) - set(X_train.columns))

# inititializing DF from projected 2019, 2020, and 2021 (& 2018) feature values
features = pd.DataFrame(np.repeat(zips_unique_that_are_also_in_target,4,axis=0))
features.columns = ['RegionName']

# adding column for year values between 2018 - 2021
seq = cycle([2018, 2019, 2020, 2021])
features['year'] = [next(seq) for count in range(features.shape[0])]

# creating subset of column names to index on
columns = []
columns += ['RegionName']
columns.extend(list(X_train.columns))

# pulling 2018 values from master DF - target
feat_2018 = target[target['year']==2018][columns]
feat_2018['year'] = 2018

# merging 2018 feature values as base to calculate 2019, 2020, and 2021 features
features = features.merge(feat_2018, on=['RegionName', 'year'], how='left')

# manipulating column names to reflect percent change values
new_columns = [x+'_diff' for x in columns]

# removing RegionName and year to strictly calc. percent changes for numerical variables
columns.remove('RegionName')
columns.remove('year')

# grouping by RegionName to calculate yearly percent change across all of the zip codes
diff = new_target.groupby('RegionName')

# percent change for each year-dependent feature
test5 = diff[columns].pct_change()

# remerging zip and year to be used when finding average percent change across each feature
test5 = pd.concat([test5, new_target[['RegionName', 'year']]], axis=1)

# percent change per zip code per year
test5 = test5.groupby('RegionName').mean()

# resetting DF index 
test5 = test5.reset_index()

# resetting columns list to include all variables (zip included)
columns = []
columns += ['RegionName']
columns.extend(list(X_train.columns))

# validating DF shapes
features_2018 = features[features['year']==2018]

# column indices to be iterated through when calculating predicted feature values
column1 = X_train.columns
column1 = column1.tolist()
column1.remove('year')

# adjusting indices to be continuous from 1 - 940
features_2018 = features_2018.reset_index()

# DF for predicted 2019 feature values
feat2_2019 = features_2018[['RegionName']]
for col in column1:
    feat2_2019[col] = features_2018[col]*test5[col] + features_2018[col]
feat2_2019['year'] = 2019

# DF for predicted 2020 feature values
# multiply mean yearly percent change by 2 to project two years from 2018
feat_2020 = features_2018[['RegionName']]
for col in column1:
    feat_2020[col] = (features_2018[col]*(test5[col]*2)) + features_2018[col]
feat_2020['year'] = 2020

# DF for predicted 2020 feature values
# multiply mean yearly percent change by 3 to project three years from 2018
feat_2021 = features_2018[['RegionName']]
for col in column1:
    feat_2021[col] = (features_2018[col]*(test5[col]*3)) + features_2018[col]
feat_2021['year'] = 2021

# resetting feat_2018 indices before concat with 2019, 2020, and 2021
feat_2018 = feat_2018.reset_index()

# concatening yearly predicted feature values into one DF
features2 = pd.concat([feat_2018, feat2_2019, feat_2020, feat_2021])
features2.drop(['Cluster', 'index'], axis=1, inplace=True)

# sorting values by zip / year to match original formatting
features2 = features2.sort_values(by=['RegionName', 'year'])

# dropping irrelevant columns with high percentage of null values
col_to_drop = ['two_parents_not_in_labor_force_families_with_young_children', 
               'two_parents_mother_in_labor_force_families_with_young_children', 'million_dollar_housing_units', 
               'vacant_housing_units_for_sale', 'father_one_parent_families_with_young_children', 
               'father_in_labor_force_one_parent_families_with_young_children', 'mobile_homes', 'armed_forces', 
               'commuters_by_subway_or_elevated', 'employed_agriculture_forestry_fishing_hunting_mining', 
               'female_female_households', 'group_quarters', 'male_male_households', 'amerindian_ratio', 
               'other_race_ratio']
features2.drop(col_to_drop, axis=1, inplace=True)

# dropping remaining rows with null values and removing 2018 feature values (not predictions)
# final version DF = features3 
features3 = features2.dropna()
feature3 = features3[features3['year']>2018]
