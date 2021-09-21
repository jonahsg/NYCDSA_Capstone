import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from copy import copy


year = ['2011','2012','2013','2014','2015','2016','2017','2018']
data_ = pd.DataFrame()

acs = pd.read_csv("../data/ACS_2011.csv")
acs['year'] = 2011

pp = pd.read_csv("../data/ACS_2012.csv")
pp['year'] = 2012
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2013.csv")
pp['year'] = 2013
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2014.csv")
pp['year'] = 2014
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2015.csv")
pp['year'] = 2015
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2016.csv")
pp['year'] = 2016
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2017.csv")
pp['year'] = 2017
acs = acs.append(pp)

pp = pd.read_csv("../data/ACS_2018.csv")
pp['year'] = 2018
data_ = acs.append(pp)

zir = pd.read_csv("../data/Zip_Zri_MultiFamilyResidenceRental.csv")


data_['geo_id'] = data_['geo_id'].astype(str).str.zfill(5)
zir['RegionName'] = zir['RegionName'].astype(str).str.zfill(5)


cols_replace = list(zir.columns)[7:]
cols_keep = list(zir.columns)[:7]
zil_new = zir.melt(id_vars=cols_keep, value_vars = cols_replace)
zil_new['variable']= pd.to_datetime(zil_new['variable'])


zil_new['year'] = zil_new['variable'].apply(lambda a: a.year)


zil_yr = zil_new.groupby(['RegionName', 'year']).mean().reset_index()
zil_yr.drop(['SizeRank','RegionID'], inplace=True, axis=1)

zil_targets = zil_yr[zil_yr['year'] > 2011]
zil_targets = zil_targets[zil_targets['year'] < 2020]


zil_yr = zil_yr.dropna()
zil_targets = zil_yr[zil_yr['year'] > 2011]
zil_yr = zil_yr[zil_yr['year'] > 2010]
zil_yr = zil_yr[zil_yr['year'] < 2019]

df_final = zil_yr.merge(data_, how='left', left_on = ['RegionName','year'], right_on=['geo_id', 'year'])

df_final.drop(['pop_5_years_over', 'speak_only_english_at_home', 'speak_spanish_at_home_low_english', 
               'pop_15_and_over', 'pop_never_married', 'pop_now_married', 'pop_separated', 'pop_widowed',
              'pop_divorced', 'geoid','speak_spanish_at_home'], axis=1, inplace=True)

df_final[df_final['geo_id'].isnull()]

df_final = df_final[df_final['RegionName']!='11249']
df_final = df_final[df_final['RegionName']!='75033']

df_final[df_final['geo_id'].isnull()]

null_zip = list(zip(df_final.columns, list(df_final.isnull().sum())))
non_zero = []
col_names = []
for i, j in null_zip:
    if j>0:
        non_zero.append(j)
        col_names.append(i)
list(zip(col_names, non_zero))

df_final_sorted = df_final.sort_values(by=['RegionName', 'year'])

null_2011 = ['associates_degree', 'bachelors_degree', 
             'high_school_diploma', 
             'less_one_year_college', \
             'masters_degree', 
             'one_year_more_college', 
             'pop_25_years_over']

null_2018 = ['white_including_hispanic', 'black_including_hispanic', 
             'amerindian_including_hispanic', 
             'asian_including_hispanic', 
             'commute_5_9_mins', 'commute_35_39_mins', 
             'commute_40_44_mins', 'commute_60_89_mins', 
             'commute_90_more_mins', 'households_retirement_income',
            'male_60_61','male_62_64']

for i in range(0, df_final_sorted.shape[0]):
    if df_final_sorted['year'].iloc[i] == 2011:
        for j in range(0,len(null_2011)):
            df_final_sorted[null_2011[j]].iloc[i] = df_final_sorted[null_2011[j]].iloc[i+1]
    if df_final_sorted['year'].iloc[i] == 2018:
        for j in range(0,len(null_2018)):
            df_final_sorted[null_2018[j]].iloc[i] = df_final_sorted[null_2018[j]].iloc[i-1]


df_final = df_final_sorted.loc[:, df_final_sorted.columns != 'do_date']
df_final = df_final.dropna()


df_final = df_final[df_final['value'] < 6000]



# Filter Cityies with less than 100,000 people
zil_city = df_final.merge(zir, on = 'RegionName', how = 'left')
zil_city = zil_city[['RegionName','City','total_pop','year']]
zil_city = zil_city[zil_city['year'] == 2018]
zil_city.drop('year', inplace = True, axis= 1)
zil_city = zil_city.groupby('City').sum('total_pop').sort_values(by = 'total_pop')

zil_filter = zil_city[zil_city['total_pop'] >= 100000].reset_index()
zil_filter = zil_filter['City']

df_filtered = df_final.merge(zir[['RegionName','City']], on = 'RegionName', how = 'left')
df_filtered = df_filtered[df_filtered['City'].isin(zil_filter)]


## Add CPI data
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
cpi = [251.645, 257.081, 263.050, 270.509, 278.802, 288.233, 297.808, 307.660]
cpi_df = pd.DataFrame()
cpi_df['year'] = year
cpi_df['cpi'] = cpi


df_addition = df_filtered.merge(cpi_df, how = 'left', on = 'year')

## Add GDP Growth (%) data
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
gdp = [1.5508, 2.2495, 1.8421, 2.526, 3.0755, 1.7114, 2.3327, 2.9965]
gdp_df = pd.DataFrame()
gdp_df['year'] = year
gdp_df['gdp'] = gdp

df_addition = df_addition.merge(gdp_df, how = 'left', on = 'year')


# Add Federal Interest Rate data
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
interest = [0.1016666667, 0.14, 0.1075, 0.08916666667, 0.1325, 0.395, 
            0.655, 1.42]
interest_df = pd.DataFrame()
interest_df['year'] = year
interest_df['interest'] = interest

df_addition = df_addition.merge(interest_df, how = 'left', on = 'year')


## Add Census business data
bus_count = pd.read_csv('../data/bus_count.csv')

bus_count['zip'] = bus_count['zip'].astype(str).str.zfill(5)
df_addition = df_addition.merge(bus_count, how = 'left', left_on = ['RegionName', 'year'], right_on = ['zip', 'year'])

# Feature Engineering
df_addition['log_value'] = np.log(df_addition['value'])


# Create Target Variable
zil_targets['year'] = zil_targets['year'] - 1

zil_targets['future_value'] = zil_targets['value']
zil_targets.drop('value', axis=1, inplace=True)


target = df_addition.merge(zil_targets, on = ['RegionName','year'], how='left')
target = target.dropna()


## More Features
# Create new columns for male ratio
target['male_ratio'] = target['male_pop']/target['total_pop']

# Drop original male / female population counts and income categories
drop1 = ['male_pop', 'female_pop', 'male_under_5', 'male_5_to_9', 
             'male_10_to_14', 'male_15_to_17', 
             'male_18_to_19', 'male_20', 
              'male_21', 'male_22_to_24', 
             'male_25_to_29', 'male_30_to_34', 
             'male_35_to_39', 'male_40_to_44', 
              'male_45_to_49', 'male_50_to_54', 
             'male_55_to_59', 'male_60_61', 'male_62_64', 'male_65_to_66', 
              'male_67_to_69', 'male_70_to_74', 'male_75_to_79', 
             'male_80_to_84', 'male_85_and_over', 
              'female_under_5', 'female_5_to_9',
             'female_10_to_14', 'female_15_to_17', 'female_18_to_19', 
              'female_20', 'female_21', 'female_22_to_24', 
             'female_25_to_29', 'female_30_to_34', 
              'female_35_to_39', 'female_40_to_44', 
             'female_45_to_49', 'female_50_to_54', 'female_55_to_59', 
              'female_60_to_61', 'female_62_to_64', 
             'female_65_to_66', 'female_67_to_69', 'female_70_to_74', 
              'female_75_to_79', 'female_80_to_84', 'female_85_and_over',
             'male_under_5', 'male_5_to_9', 'male_10_to_14', 'male_15_to_17', 
             'male_18_to_19', 'male_20', 
             'male_21', 'male_22_to_24', 'male_25_to_29','male_30_to_34',
             'male_35_to_39','male_40_to_44', 
             'male_45_to_49', 'male_50_to_54', 
             'male_55_to_59', 'male_60_61', 'male_62_64', 
             'male_65_to_66', 
             'male_67_to_69', 'male_70_to_74', 'male_75_to_79', 
             'male_80_to_84', 'male_85_and_over', 
             'female_under_5', 'female_5_to_9', 'female_10_to_14', 
             'female_15_to_17', 'female_18_to_19', 
             'female_20', 'female_21', 'female_22_to_24', 
             'female_25_to_29', 'female_30_to_34', 'female_35_to_39', 
             'female_40_to_44', 'female_45_to_49',
             'female_50_to_54', 'female_55_to_59', 
             'female_60_to_61', 
             'female_62_to_64', 'female_65_to_66', 
             'female_67_to_69', 'female_70_to_74', 
             'female_75_to_79', 
             'female_80_to_84', 'female_85_and_over',
             'income_less_10000', 'income_10000_14999', 'income_15000_19999', 
             'income_20000_24999', 'income_25000_29999', 'income_30000_34999', 
             'income_35000_39999', 'income_40000_44999', 
             'income_45000_49999', 'income_50000_59999', 
             'income_60000_74999', 'income_75000_99999', 
             'income_100000_124999', 'income_125000_149999', 
             'income_150000_199999', 'income_200000_or_more']
target.drop(drop1, axis=1, inplace=True)

# Create new columns for demographic ratios
target['white_ratio'] = target['white_pop']/target['total_pop']

target['black_ratio'] = target['black_pop']/target['total_pop']

target['asian_ratio'] = target['asian_pop']/target['total_pop']

target['hispanic_ratio'] = target['hispanic_pop']/target['total_pop']

target['amerindian_ratio'] = target['amerindian_pop']/target['total_pop']

target['other_race_ratio'] = target['other_race_pop']/target['total_pop']


# Drop columns for demographic that aren't needed anymore and misc columns
drop2 = ['white_pop', 'black_pop', 'asian_pop', 'hispanic_pop', 'amerindian_pop', 'other_race_pop', 
         'two_or_more_races_pop', 'not_hispanic_pop',
         'asian_male_45_54', 'asian_male_55_64', 'black_male_45_54',
         'black_male_55_64', 'hispanic_male_45_54', 
         'hispanic_male_55_64', 'white_male_45_54', 'white_male_55_64',
         'male_45_64_associates_degree', 
         'male_45_64_bachelors_degree', 'male_45_64_graduate_degree',
         'male_45_64_less_than_9_grade', 
         'male_45_64_grade_9_12', 'male_45_64_high_school', 
         'male_45_64_some_college', 'male_45_to_64']
target.drop(drop2, axis=1, inplace=True)


# Create employement ratio and drop population counts
target['employed_ratio'] = target['employed_pop']/target['total_pop']

target.drop(['employed_pop', 'unemployed_pop'], axis=1, inplace=True)

# Drop some other misc features
drop3 = ['pop_16_over', 'pop_in_labor_force', 'pop_25_64',
       'pop_determined_poverty_status', 'population_1_year_and_over',
       'population_3_years_over', 'pop_25_years_over',
       'four_more_cars', 'no_car', 'no_cars', 'one_car', 
       'two_cars', 'three_cars', 'two_parents_not_in_labor_force_families_with_young_children', 
       'two_parents_mother_in_labor_force_families_with_young_children', 'million_dollar_housing_units', 
       'vacant_housing_units_for_sale', 'father_one_parent_families_with_young_children', 
       'father_in_labor_force_one_parent_families_with_young_children', 'mobile_homes', 'armed_forces', 
       'commuters_by_subway_or_elevated', 'employed_agriculture_forestry_fishing_hunting_mining', 
       'female_female_households', 'group_quarters', 'male_male_households', 'amerindian_ratio', 
       'other_race_ratio']
target.drop(drop3, axis=1, inplace=True)

target['civil_labor_ratio'] = target['civilian_labor_force']/target['total_pop']
target.drop('civilian_labor_force', axis=1, inplace=True)




## Cluster on three features
df_small_fil = target[['value', 'total_pop', 'median_income']]

scaler = StandardScaler().fit(df_small_fil)
features = scaler.transform(df_small_fil)
df_scal = pd.DataFrame(features, columns = df_small_fil.columns)


columns = df_small_fil.columns
kmeans = KMeans(n_clusters = 3)
y = kmeans.fit_predict(df_small_fil[columns])
   
target['Cluster'] = y



# Split Train and Test
train = target[target['year'] < 2018]
test = target[target['year'] == 2018]

train_feat = train.drop('future_value', axis=1)
test_feat = test.drop('future_value', axis=1)


test_feat = test_feat.drop('value', axis=1)
train_feat = train_feat.drop('value', axis=1)


Y_train = np.log(train[['future_value']])
X_train = train_feat.select_dtypes(exclude=['object'])

Y_test = np.log(test[['future_value']])
X_test = test_feat.select_dtypes(exclude=['object'])



# Split into Clusters
Y_ctrain = Y_train.copy()
Y_ctrain['Cluster'] = X_train['Cluster']
Y_ctest = Y_test.copy()
Y_ctest['Cluster'] = X_test['Cluster']


Y_c1train = Y_ctrain[Y_ctrain['Cluster'] == 0]
Y_c1train.drop('Cluster', axis=1, inplace=True)
Y_c2train = Y_ctrain[Y_ctrain['Cluster'] == 1]
Y_c2train.drop('Cluster', axis=1, inplace=True)
Y_c3train = Y_ctrain[Y_ctrain['Cluster'] == 2]
Y_c3train.drop('Cluster', axis=1, inplace=True)

Y_c1test = Y_ctest[Y_ctest['Cluster'] == 0]
Y_c1test.drop('Cluster', axis=1, inplace=True)
Y_c2test = Y_ctest[Y_ctest['Cluster'] == 1]
Y_c2test.drop('Cluster', axis=1, inplace=True)
Y_c3test = Y_ctest[Y_ctest['Cluster'] == 2]
Y_c3test.drop('Cluster', axis=1, inplace=True)



c1_train = X_train[X_train['Cluster'] == 0]
c2_train = X_train[X_train['Cluster'] == 1]
c3_train = X_train[X_train['Cluster'] == 2]


c1_test = X_test[X_test['Cluster'] == 0]
c2_test = X_test[X_test['Cluster'] == 1]
c3_test = X_test[X_test['Cluster'] == 2]


