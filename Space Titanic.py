# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:43:36 2023

@author: Nevs
"""
#%% Import dependency modules
import os
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

#%% Initial data import

# Read data from csvs

train_dt = pd.read_csv(
    r'C:\Users\Nevs\Documents\Kaggle\Interview\Spaceship Titanic\train.csv')

test_dt = pd.read_csv(
    r'C:\Users\Nevs\Documents\Kaggle\Interview\Spaceship Titanic\test.csv')

#%% helper functions

def ex_nan(df, column):
    '''
    
    Parameters
    ----------
    df : pandas DataFrame
        parent Dataframe onto which the passed column is checked for nan vals.
    column : str
        column from which nans are to be excluded.

    Returns
    -------
    pandas Series obj
        the column from parent dataframe is return sans nan vals.

    '''
    
    return df[column][pd.notna(df[column])]

def pct_na(df, return_df = 0):
    '''
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to be tested with some na rows.
    return_df : int, optional
        choose whether to return DataFrame or pct. The default is 0.

    Returns
    -------
    Float
        percentage of rows with at least one na column.
    DataFrame
        DF object of containing na values 

    '''
    missing_df = df[df.isna().any(axis=1)]
    
    if return_df == 0:
        return len(missing_df) / len(df)
    elif return_df == 1:
        return missing_df
    
def reverse_dict(dictionary):
    '''
    

    Parameters
    ----------
    dictionary : dict
        dictionary to be inverted.

    Returns
    -------
    dict
        inverted dict to one supplied.

    '''
    return {k: v for v, k in dictionary.items()}


def fix_dummies_cols(train_dataset, test_dataset):
    '''

    Parameters
    ----------
    train_dataset : DataFrame
        training dataset where categorical variables are pivotted wider,
        contains full set of columns.
    test_dataset : DataFrame
        testing dataset where categorical variables are pivotted wider,
        may not include certain columns from full set of variables.

    Returns
    -------
    DataFrame
        testing dataset containing full set of columns.

    '''
    
    train_dataset = x_train_var_dummies.copy()
    test_dataset = x_test_var.copy()
    
    for i in train_dataset.columns:
        if i not in test_dataset.columns:
            test_dataset[i] = np.uint8(0)
    
    return test_dataset[train_dataset.columns]
    


#%% Exploratory data analysis

#general idea on col d types
train_dt.infer_objects()
train_dt.info()

test_dt.infer_objects()
test_dt.info()


### determine number of rows with any missing data
print(f'{round(pct_na(train_dt)*100,2)}% of data missing at outset in train')
      #approx  24%
      
print(f'{round(pct_na(test_dt)*100,2)}% of data missing at outset in test')
     #approx  23%

#approx 24% of rows in training have missing vals, decide not to delete na rows

###Here make assumption that training and test data comprise the full 
### population information, thus filling na and data analysis is not
### compromised by joining two datasets

#set temp known val for test_dt transported
test_dt['Transported'] = 'N.a'

#save copy of dfs if it requires referencing
train_dt_orig = train_dt.copy()
test_dt_orig = test_dt.copy()

#join train and test datasets
train_dt = pd.concat([train_dt, test_dt])



### initial data clean from descriptions on kaggle

#set the 'PassengerId' variable as the index for both datasets
train_dt = train_dt.set_index('PassengerId')

#drop Name col
train_dt.drop('Name', axis=1, inplace= True)

#split 'Cabin' variable into 'deck' 'room_num', 'side'
train_dt[
    ['deck',
     'room_num',
     'side']
    ] = train_dt['Cabin'].str.split(r'/', expand = True)

#fix room_num as numerical type
train_dt['room_num'] = train_dt['room_num'].astype(float)

#fix Cryo_sleep as numerical type
train_dt['CryoSleep'] = train_dt['CryoSleep'].replace({True:1, False:0})


#fix VIP as numerical type
train_dt['VIP'] = (train_dt['VIP'] == True).replace({True:1, False:0})

#drop Cabin col
train_dt.drop('Cabin', axis=1, inplace= True)






#determine the dependent variables columns
cols = train_dt.columns.drop('Transported')

#Categorical Variable vs independent
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
            'deck', 'side']

#Numerical Variable vs independent
num_cols = [x for x in cols if x not in cat_cols]
#num_cols.remove('PassengerId')




#for each dep var, find the unique row count
unique_vals = {k: v for
               k, v in zip(cols,
                                 [train_dt[x].value_counts()
                                  for x in cols])}


#setup dummy train_dt dataset for categorical analysis
cat_v_label_train_dt = train_dt.iloc[:len(train_dt_orig),:].copy()

cat_v_label_train_dt['Transported'] = cat_v_label_train_dt[
    'Transported'].astype(int)

#mean of the entire training set for the transported value
training_set_mean = cat_v_label_train_dt['Transported'].mean()

#Cat variables influence on dep var
cat_v_label = {k:v for
               k,v in zip(cat_cols,
                          [cat_v_label_train_dt[
                              [str(x), 'Transported']
                              ].groupby(str(x)).mean() for x in cat_cols]
                          )}




#%% fill na's


#designate amenities variables as service cost
service_cost =['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

#postulate service cost variables for cryosleep passengers is zero
# sum full dataset to see if this holds, if so, fill na vals with zero
cryo_cost_test = train_dt[train_dt['CryoSleep'] == 1][service_cost].sum(
    skipna = True)

#since above hypothesis holds, we fill sleeping passengers service cost as 0
# setup a few dummy vectors and dataframes to hold correct conditions
bool_vect1 = pd.DataFrame(train_dt.columns.isin(service_cost))
bool_vect2 = pd.DataFrame((train_dt['CryoSleep'] == 1))

# use matrix operation and python bool handling to setup dummy matrix
bool_mat = pd.DataFrame(np.dot(bool_vect2, bool_vect1.transpose()),
                        index = train_dt.index,
                        columns = train_dt.columns)

# use dummy matrix to replace all cryosleep service_cost with 0
train_dt = train_dt.mask(bool_mat, 0)

print(f' {round(pct_na(train_dt)*100,2)}% of rows contain a na value after \
a context based fill')
#now 16.5% of data with missing vals



### Initially using passenger grouping to fill na

#dataframe of used to house grouping operations
ffill_train_dt = train_dt.copy()

#setup grouping variable
ffill_train_dt['PassengerGroup'] = [i.split('_')[0] for i in
                                    ffill_train_dt.index]

#for in passengers in the same passenger group, forwardfill all missing data
ffill_train_dt = ffill_train_dt.groupby('PassengerGroup').fillna(method =
                                                                 'ffill')


group_sensible_cols = ['HomePlanet', 'CryoSleep', 'Destination',
                       'VIP', 'deck', 'room_num', 'side']

#replace original train_dt columns with ffilled cols where it would make sense
train_dt[group_sensible_cols] = ffill_train_dt[group_sensible_cols]






print(f' {round(pct_na(train_dt)*100,2)}% of rows contain a na value after \
a groupby fill')
#now 14% of data with missing vals





#Histogram of combined service cost combined pre fill
combined_vals = train_dt[train_dt['CryoSleep'] == 0][service_cost].sum(
    axis = 1, skipna = True)

plt.hist(combined_vals,
         bins = round(len(train_dt)/100))
plt.title('distribution of combined service cost')
plt.show()

### Now attempt to understand amenities variables

# =============================================================================
# #designate amenities variables as service cost, remove from num_cols
# service_cost =['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# =============================================================================

#remove service cost vars from num_cols
num_cols = [x for x in num_cols if x not in service_cost]

# hypothesize service cost variable can be combined, check distributions
for x in service_cost:
    plt.hist(ex_nan(train_dt[train_dt['CryoSleep'] == 0], x),
             bins = round(len(train_dt)/100))
    plt.title(f'distribution of {x}')
    plt.show()
    
#fill missing service cost vars with average of inner 95% of data (ex outliers)
for x in service_cost:
    #tmp_ave = train_dt[x].mean()
    
    tmp_ave = train_dt[x][
        (train_dt[x] <= train_dt[
            train_dt['CryoSleep'] == 0
            ][x].quantile(0.975)) &
        (train_dt[x] >= train_dt[
            train_dt['CryoSleep'] == 0
            ][x].quantile(0.025))
        ].mean()
    
    train_dt[x] = train_dt[x].fillna(value = tmp_ave)
    
#Histogram of combined service cost 
combined_vals = train_dt[train_dt['CryoSleep'] == 0][service_cost].sum(
    axis = 1)

plt.hist(combined_vals,
         bins = round(len(train_dt)/100))
plt.title('distribution of combined service_cost post fill')
plt.show()


print(f' {round(pct_na(train_dt)*100,2)}% of rows contain a na value after \
service cost filled with mean of inner 95% values')
#now 8.03% of data with missing vals







### fill na on categorical values, with values that make intuitive sense

#fill na's
    #planet based vars filled with a placeholder planet
train_dt['HomePlanet'] = train_dt['HomePlanet'].fillna(value = 'Undef_planet')
train_dt['Destination'] = train_dt['Destination'].fillna(value= 'Undef_planet')
    
    #vars with states that are exclusive, set to excluded state
    # additional benefit is that these are also mean states 
train_dt['CryoSleep'] = train_dt['CryoSleep'].fillna(value = 0)
train_dt['VIP'] = train_dt['VIP'].fillna(value = 0)


#check all Destinations equal to HomePlanet
home_and_away = train_dt[train_dt['Destination'] == train_dt['HomePlanet']]



print(f' {round(pct_na(train_dt)*100,2)}% of rows contain a na value after \
arbitrary filling')
#now 3.67% of data with missing vals















### synthetically predict remaining na values


# only 'Age' and cabin split variables have na vals
train_dt_age_and_cabin_na = pct_na(train_dt,1)



#check number of rows of na's for each, if == total n.o. rows then there are
# no rows with na's in both 'Age' and cabin split variables
len(train_dt_age_and_cabin_na[train_dt_age_and_cabin_na['Age'].isna()]) + \
len(train_dt_age_and_cabin_na[train_dt_age_and_cabin_na['deck'].isna()]) - \
len(train_dt_age_and_cabin_na)

#only four rows have a combination, drop them from missing dt
train_dt_age_or_cabin_na = train_dt_age_and_cabin_na[
    ~((train_dt_age_and_cabin_na['Age'].isna()) &
    (train_dt_age_and_cabin_na['deck'].isna()))]

#setup x,y test for prediction variables with na value, remove target col
train_dt_testingset_val_pred_xy = pct_na(train_dt,
                                         1).drop(
                                             ['Transported'],axis=1)
                                             

#setup x,y training vars
train_dt_val_pred_xy = train_dt[
    ~(train_dt.index.isin(
        train_dt_testingset_val_pred_xy.index
        ))].drop(['Transported'], axis=1)



#convert Cryosleep & VIP to categorical format
train_dt_testingset_val_pred_xy['CryoSleep'] = train_dt_testingset_val_pred_xy[
    'CryoSleep'].replace({1:'A', 0:'B'})

train_dt_val_pred_xy['VIP'] = train_dt_val_pred_xy[
    'VIP'].replace({1:'A', 0:'B'})




#seperate model for age, and room_num
# as they can be thought of as a numerical variable
na_val_pred_modl_age = LinearRegression()


#models for deck vars, as they are similar to cat variables
na_val_pred_modl_deck = KNeighborsClassifier(n_neighbors=5)


#model for side var, 0,1 makes sense to use logistic regression
na_val_pred_modl_side = LogisticRegression()



#setup training data excluding the rows with na in any of the prediction vars
x_train_var = train_dt_val_pred_xy.loc[
    :,~train_dt_val_pred_xy.columns.isin(
        ['Age', 'deck', 'room_num', 'side'])]

#convert categorical vars into indicators
x_train_var_dummies = pd.get_dummies(x_train_var)



#create dictionary housing predicted variables
pred_vals_dict = {}

### begin prediction of na vals on 'Age' 'deck, 'room_num' and 'side'

    #Age prediction
        
#testing dataset for age pred var
x_test_var = train_dt_testingset_val_pred_xy[
    train_dt_testingset_val_pred_xy['Age'].isna()].drop(columns = [
        'Age', 'deck', 'room_num', 'side'])

#convert to wide format
x_test_var = pd.get_dummies(x_test_var)

#certain columns are missing from x_test_var as the number of testing entries
# may not have the full scope of unique states on every variable
x_test_var = fix_dummies_cols(x_train_var_dummies, x_test_var)
    
#create y training for 'Age', a new y training to be created for each pred var
y_train_var = train_dt_val_pred_xy['Age']

#regress Age vs filled variables excluding 'Transported'
na_val_pred_modl_age.fit(x_train_var_dummies, y_train_var)


#into dict, place predicted 'Age' values
pred_vals_dict['Age'] = pd.DataFrame(
    {'Age': np.round(na_val_pred_modl_age.predict(x_test_var), 0)},
    index = x_test_var.index)







    #room_num prediction
#testing dataset for deck, room_num, side pred vars
x_test_var = train_dt_testingset_val_pred_xy[
    train_dt_testingset_val_pred_xy['deck'].isna()].drop(columns = [
        'Age','deck', 'room_num', 'side'])

#convert to wide format
x_test_var = pd.get_dummies(x_test_var)

#certain columns are missing from x_test_var as the number of testing entries
# may not have the full scope of unique states on every variable
x_test_var = fix_dummies_cols(x_train_var_dummies, x_test_var)


#create y training for 'room_num', similar to 'Age'
y_train_var = train_dt_val_pred_xy['room_num']

#regress room_num vs filled variables excluding 'Transported'
na_val_pred_modl_age.fit(x_train_var_dummies, y_train_var)


#into dict, place predicted 'room_num' values
pred_vals_dict['room_num'] = pd.DataFrame(
    {'room_num': np.round(na_val_pred_modl_age.predict(x_test_var), 0)},
    index = x_test_var.index)







    #deck prediction
#mapping dictionary for ease
deck_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}

#create y training for 'deck', similar to 'Age', change: cat to num var type
y_train_var = train_dt_val_pred_xy['deck'].map(deck_map)



#regress room_num vs filled variables excluding 'Transported'
na_val_pred_modl_deck.fit(x_train_var_dummies, y_train_var)


#into dict, place predicted 'room_num' values
pred_vals_dict['deck'] = pd.DataFrame(
    {'deck': np.round(na_val_pred_modl_deck.predict(x_test_var), 0)},
    index = x_test_var.index)

#into dict, place predicted 'deck' values
pred_vals_dict['deck'] = pd.DataFrame(pred_vals_dict['deck']['deck'].map(
    reverse_dict(deck_map)))







    #side prediction
#mapping dictionary for ease
side_map = {'P':0, 'S':1}

#create y training for 'side', similar to 'deck'
y_train_var = train_dt_val_pred_xy['side'].map(side_map)



#regress room_num vs filled variables excluding 'Transported'
na_val_pred_modl_side.fit(x_train_var_dummies, y_train_var)


#into dict, place predicted 'room_num' values
pred_vals_dict['side'] = pd.DataFrame(
    {'side': np.round(na_val_pred_modl_side.predict(x_test_var), 0)},
    index = x_test_var.index)

#into dict, place predicted 'side' values
pred_vals_dict['side'] = pd.DataFrame(pred_vals_dict['side']['side'].map(
    reverse_dict(side_map)))







### place regression imputed values back into train_dt set

#iterate over all imputed vals in predicted dictionary, fit back into train_dt
for k,v in pred_vals_dict.items():
    for i in v.index:
        train_dt.loc[i, k] = v.loc[i][0]
    

print(f' {round(pct_na(train_dt)*100,2)}% of rows contain a na value after \
imputing missing values using modelling techniques')
#percentage of rows with na vals is now equal to 0


### data has now been cleaned, next step is to prep data for use in models






#%%data prep


#combine service cost variables
train_dt['ServiceCost'] = train_dt['RoomService'] + train_dt['FoodCourt'] \
    + train_dt['ShoppingMall'] + train_dt['Spa'] + train_dt['VRDeck']
    
#add new col to numerical col list
num_cols.append('ServiceCost')
    
#drop individual service cost variables
train_dt.drop(columns = service_cost, inplace = True)

#convert Cryosleep & VIP to categorical format
train_dt['CryoSleep'] = train_dt['CryoSleep'].replace({1:'A', 0:'B'})
train_dt['VIP'] = train_dt['VIP'].replace({1:'A', 0:'B'})
 

#seperate test_dt back out of train_dt
test_dt = train_dt[train_dt['Transported'] == 'N.a'].copy().drop(columns = 
                                                                 'Transported')
#remove test_dt rows from train_dt
train_dt = train_dt[train_dt['Transported'] != 'N.a'].copy()






#%% apply preprocessing 

### pre-format data
#seperate into indep and dep vars for training set
x_training_dt = train_dt.drop(columns = 'Transported').copy()

y_training_dt = train_dt['Transported'].copy().astype(int)

#assign testing set with correct var name (for ease)
x_test_dt = test_dt.copy()


### apply preprocessing techniques, scaling numerical, widen categorical

#pivot wider categorical vars to suit models
x_training_dt_wider = pd.get_dummies(x_training_dt)
x_test_dt_wider = pd.get_dummies(x_test_dt)


#init scalar
scale = StandardScaler()

#apply to numerical cols
x_training_dt_wider[num_cols] = scale.fit_transform(
    x_training_dt_wider[num_cols])

x_test_dt_wider[num_cols] = scale.fit_transform(
    x_test_dt_wider[num_cols])

#%% Modelling

#Initialize models
LogReg = LogisticRegression()
KNN = KNeighborsClassifier()
SupVectMach = svm.SVC()

#train models on training data
LogReg.fit(x_training_dt_wider, y_training_dt)
KNN.fit(x_training_dt_wider, y_training_dt)
SupVectMach.fit(x_training_dt_wider, y_training_dt)

#predict our target variable using the models
y_pred_LogReg = LogReg.predict(x_test_dt_wider)
y_pred_KNN = KNN.predict(x_test_dt_wider)
y_pred_SupVectMach = SupVectMach.predict(x_test_dt_wider)

#attach results into list in DataFrame format
results = [pd.DataFrame(y_pred_LogReg, index=test_dt.index,
                        columns = ['LogReg']),
           pd.DataFrame(y_pred_KNN, index=test_dt.index,
                        columns = ['KNN']),
           pd.DataFrame(y_pred_SupVectMach, index=test_dt.index,
                        columns = ['SVM'])]

#interperate results as bool dtype
results = [i.astype(bool) for i in results]

#directory to export
os.chdir(r'C:\Users\Nevs\Documents\Kaggle\Interview\Spaceship Titanic\\')


# =============================================================================
# ### only run when you would like to export results
# for i in results:
#     tech = i.columns[0]
#     i.columns = ['Transported']
#     i.to_csv(f'{tech}.csv')
# =============================================================================


'''
#%% Application of GridSearch CV

#split data using test train split
X_train, X_test, y_train, y_test = train_test_split(
    x_training_dt_wider,
    y_training_dt,
    test_size = 0.2,
    random_state = 828)

#import gridsearch mod
from sklearn.model_selection import GridSearchCV

#complimentary ml algos to our original ml set
LogReg_GS = LogisticRegression()
KNN_GS = KNeighborsClassifier()
SupVectMach_GS = svm.SVC()





### begin LogReg optimization

para_LogR_GS = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'penalty':['none', 'elasticnet', 'l1', 'l2'],
                'C':[0.001, 0.01, 0.1, 1, 10, 100]}


LogR_gd_search = GridSearchCV(estimator = LogReg_GS,
                         param_grid= para_LogR_GS,
                         n_jobs=(5), verbose = 1)

LogR_gd_search.fit(X_train, y_train)

LogR_gd_search.best_params_

OptLogReg = LogisticRegression(C =
                               LogR_gd_search.best_params_['C'],
                               penalty = 
                               LogR_gd_search.best_params_['penalty'],
                               solver = 
                               LogR_gd_search.best_params_['solver'])
                               
OptLogReg.fit(x_training_dt_wider, y_training_dt)












### begin KNN optimization
para_KNN_GS = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
               'weights':['uniform', 'distance'],
               'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
               'p': [1,2]}


KNN_gd_search = GridSearchCV(estimator = KNN_GS,
                         param_grid= para_KNN_GS,
                         n_jobs=(5), verbose = 1)

KNN_gd_search.fit(X_train, y_train)

KNN_gd_search.best_params_

OptKNN = KNeighborsClassifier(algorithm =
                              KNN_gd_search.best_params_['algorithm'],
                              n_neighbors = 
                              KNN_gd_search.best_params_['n_neighbors'],
                              p = 
                              KNN_gd_search.best_params_['p'],
                              weights = 
                              KNN_gd_search.best_params_['weights'])

                               
OptKNN.fit(x_training_dt_wider, y_training_dt)







### begin SVM optimization
para_SVC_GS = {'C':[0.001, 0.01, 0.1, 1, 10]}


SVC_gd_search = GridSearchCV(estimator = SupVectMach_GS,
                             param_grid= para_SVC_GS,
                             n_jobs=(5), verbose = 1)

SVC_gd_search.fit(X_train, y_train)

SVC_gd_search.best_params_

OptSVC = svm.SVC(C= SVC_gd_search.best_params_['C'])

                               
OptSVC.fit(x_training_dt_wider, y_training_dt)
















Pred_OptLogReg = OptLogReg.predict(x_test_dt_wider)
Pred_OptKNN = OptKNN.predict(x_test_dt_wider)
Pred_OptSVC = OptSVC.predict(x_test_dt_wider)




#attach results into list in DataFrame format
results_opt = [pd.DataFrame(Pred_OptLogReg, index=test_dt.index,
                            columns = ['LogReg']),
               pd.DataFrame(Pred_OptKNN, index=test_dt.index,
                            columns = ['KNN']),
               pd.DataFrame(Pred_OptSVC, index=test_dt.index,
                            columns = ['SVM'])]

#interperate results as bool dtype
results_opt = [i.astype(bool) for i in results]

#directory to export
os.chdir(r'C:\Users\Nevs\Documents\Kaggle\Interview\Spaceship Titanic\\')



### only run when you would like to export results
for i in results_opt:
    tech = i.columns[0]
    i.columns = ['Transported']
    i.to_csv(f'{tech}_opt.csv')

'''









