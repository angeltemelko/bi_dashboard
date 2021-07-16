from sklearn.metrics import r2_score, mean_absolute_error
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import autocorrelation_plot
import pmdarima as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from data_preparation import USA_vaccination_data, USA_cases_data, USA_deaths_data, USA_icu_data, USA_R_data
from data_preparation import FRA_vaccination_data, FRA_cases_data, FRA_deaths_data, FRA_icu_data, FRA_R_data

"""
Vaccinations
"""
# =============================================================================
# AutoRegression USA vaccination
# =============================================================================
split= int(0.8* len(USA_vaccination_data))

# split split the data in full series, train series and test series
full_series= USA_vaccination_data[['people_fully_vaccinated_per_hundred']]
train_series= full_series.iloc[:split]
test_series= full_series.iloc[split:]

R2_best= -np.inf
# Build a model for each lag value within the range 1:30, evaluate with R-squared
for lag in range(1, 30):
    model= AutoReg(endog= train_series, lags= lag, old_names= False).fit()
    
    Y_pred= model.predict(start=len(train_series), end=len(full_series)-1)
    
    MAE= mean_absolute_error(test_series, Y_pred)
    R2= r2_score(test_series, Y_pred)
    if R2> R2_best:
        # Only save and report back R2 and MAE if next model performs better than previous best
        USA_vac_lag= lag
        R2_best= R2
        print(lag, "R2 \t:\t", R2)
        print(lag, "MAE \t:\t", MAE) 

# Use the best lag-value to build the model with the full data.
vaccine_model_USA= AutoReg(endog= full_series, lags= USA_vac_lag, old_names= False).fit()

# write the model to a .sav file so it does not have to be run again in further use
filename = 'models/vaccine_model_USA.sav'
pickle.dump(vaccine_model_USA, open(filename, 'wb'))
# =============================================================================
# AutoRegression FRA vaccination
# =============================================================================
split= int(0.8* len(FRA_vaccination_data))
full_series= FRA_vaccination_data[['people_fully_vaccinated_per_hundred']]
train_series= full_series.iloc[:split]
test_series= full_series.iloc[split:]

R2_best= -np.inf
MAE_best= np.inf
for lag in range(1, 30):
    model= AutoReg(endog= train_series, lags= lag, old_names= False).fit()
    
    Y_pred= model.predict(start=len(train_series), end=len(full_series)-1)
    
    MAE= mean_absolute_error(test_series, Y_pred)
    R2= r2_score(test_series, Y_pred)
    if R2> R2_best:
        # Only save and report back R2 and MAE if next model performs better than previous best
        FRA_vac_lag= lag
        R2_best= R2
        print(lag, "R2 \t:\t", R2)
        print(lag, "MAE \t:\t", MAE) 

# Use the best lag-value to build the model with the full data.
vaccine_model_FRA= AutoReg(endog= full_series, lags= FRA_vac_lag, old_names= False).fit()

# write the model to a .sav file so it does not have to be run again in further use
filename = 'models/vaccine_model_FRA.sav'
pickle.dump(vaccine_model_FRA, open(filename, 'wb'))
# =============================================================================
# Auto-ARIMA USA cases
# =============================================================================
print("\n USA cases daily")

split= int(0.8* len(USA_cases_data))

full_series= USA_cases_data[['new_cases_per_million']]
train_values= USA_cases_data[['new_cases_per_million']].iloc[:split]

# create training sets
Y_train= full_series.iloc[:split]
X_train= USA_cases_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'new_cases_per_million':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= full_series[['new_cases_per_million']].iloc[split:]
X_test= USA_cases_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'new_cases_per_million':[np.sqrt(value) for value in Y_test.iloc[:,0]]})

# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
cases_model_USA= pm.auto_arima(Y_train_trans, X_train)
filename = 'models/cases_model_USA.sav'
pickle.dump(cases_model_USA, open(filename, 'wb'))
# =============================================================================
# Auto-ARIMA FRA cases
# =============================================================================
print("\n France cases daily")

split= int(0.8* len(FRA_cases_data))

full_series= FRA_cases_data[['new_cases_per_million']]
train_values= FRA_cases_data[['new_cases_per_million']].iloc[:split]

# create training sets
Y_train= full_series.iloc[:split]
X_train= FRA_cases_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'new_cases_per_million':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= full_series[['new_cases_per_million']].iloc[split:]
X_test= FRA_cases_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'new_cases_per_million':[np.sqrt(value) for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
cases_model_FRA= pm.auto_arima(Y_train, X_train)
filename = 'models/cases_model_FRA.sav'
pickle.dump(cases_model_FRA, open(filename, 'wb'))
# =============================================================================
# Auto-ARIMA USA deaths total
# =============================================================================
print("\n USA deaths total")

split= int(0.8* len(USA_deaths_data))

full_series= USA_deaths_data[['total_deaths']]

# create training sets
Y_train= USA_deaths_data[['total_deaths']].iloc[:split]
X_train= USA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'total_deaths':[value**2 for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= USA_deaths_data[['total_deaths']].iloc[split:]
X_test= USA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'total_deaths':[value**2 for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", np.sqrt(mean_absolute_error(Y_test_trans, Y_pred)))
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", np.sqrt(mean_absolute_error(Y_test_trans, Y_pred)))
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
total_deaths_model_USA= pm.auto_arima(Y_train, X_train)
filename = 'models/total_deaths_model_USA.sav'
pickle.dump(total_deaths_model_USA, open(filename, 'wb'))

# =============================================================================
# Auto-ARIMA FRA deaths total
# =============================================================================
print("\n France deaths total")

split= int(0.8* len(FRA_deaths_data))

full_series= FRA_deaths_data[['total_deaths']]

# create training sets
Y_train= FRA_deaths_data[['total_deaths']].iloc[:split]
X_train= FRA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'total_deaths':[value**2 for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= FRA_deaths_data[['total_deaths']].iloc[split:]
X_test= FRA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'total_deaths':[value**2 for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", np.sqrt(mean_absolute_error(Y_test_trans, Y_pred)))
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", np.sqrt(mean_absolute_error(Y_test_trans, Y_pred)))
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
total_deaths_model_FRA= pm.auto_arima(Y_train, X_train)
filename = 'models/total_deaths_model_FRA.sav'
pickle.dump(total_deaths_model_FRA, open(filename, 'wb'))

# =============================================================================
# Auto-ARIMA USA deaths daily
# =============================================================================
print("\n USA deaths daily")

split= int(0.8* len(USA_deaths_data))

full_series= USA_deaths_data[['new_deaths_per_million']]

# create training sets
Y_train= USA_deaths_data[['new_deaths_per_million']].iloc[:split]
X_train= USA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'new_deaths_per_million':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= USA_deaths_data[['new_deaths_per_million']].iloc[split:]
X_test= USA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'new_deaths_per_million':[np.sqrt(value) for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
new_deaths_model_USA= pm.auto_arima(Y_train_trans)
filename = 'models/new_deaths_model_USA.sav'
pickle.dump(new_deaths_model_USA, open(filename, 'wb'))

# =============================================================================
# Auto-ARIMA FRA deaths daily
# =============================================================================
print("\n France deaths daily")

split= int(0.8* len(FRA_deaths_data))

full_series= FRA_deaths_data[['new_deaths']]

# create training sets
Y_train= FRA_deaths_data[['new_deaths']].iloc[:split]
X_train= FRA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'new_deaths':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= FRA_deaths_data[['new_deaths']].iloc[split:]
X_test= FRA_deaths_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'new_deaths':[np.sqrt(value) for value in Y_test.iloc[:,0]]})

# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
new_deaths_model_FRA= pm.auto_arima(Y_train_trans, X_train)
filename = 'models/new_deaths_model_FRA.sav'
pickle.dump(new_deaths_model_FRA, open(filename, 'wb'))
# =============================================================================
# Auto-ARIMA USA icu
# =============================================================================
print("\n USA icu")

split= int(0.8* len(USA_icu_data))

full_series= USA_icu_data[['total_icu_per_million']]

# create training sets
Y_train= USA_icu_data[['total_icu_per_million']].iloc[:split]
X_train= USA_icu_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'total_icu_per_million':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= USA_icu_data[['total_icu_per_million']].iloc[split:]
X_test= USA_icu_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'total_icu_per_million':[np.sqrt(value) for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
icu_model_USA= pm.auto_arima(Y_train_trans)
filename = 'models/icu_model_USA.sav'
pickle.dump(icu_model_USA, open(filename, 'wb'))
# =============================================================================
# Auto-ARIMA FRA icu
# =============================================================================
print("\n France icu")

split= int(0.8* len(FRA_icu_data))

full_series= FRA_icu_data[['total_icu_per_million']]

# create training sets
Y_train= FRA_icu_data[['total_icu_per_million']].iloc[:split]
X_train= FRA_icu_data[['people_fully_vaccinated_per_hundred']].iloc[:split]
# transform training set using appropriate transformation
Y_train_trans= pd.DataFrame({'total_icu_per_million':[np.sqrt(value) for value in Y_train.iloc[:,0]]})

# create test sets
Y_test= FRA_icu_data[['total_icu_per_million']].iloc[split:]
X_test= FRA_icu_data[['people_fully_vaccinated_per_hundred']].iloc[split:]
# transform test set using appropriate transformation
Y_test_trans= pd.DataFrame({'total_icu_per_million':[np.sqrt(value) for value in Y_test.iloc[:,0]]})


# build simple ARIMA model using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train)
Y_pred= model.predict(n_periods= len(Y_test))
print("mean_absolute_error Simple: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Simple: ", r2_score(Y_test, Y_pred))

# build simple ARIMA model using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans)
Y_pred= model.predict(n_periods= len(Y_test_trans))
print("mean_absolute_error Simple transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Simple transformed: ", r2_score(Y_test_trans, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima and print MAE and R-squared scores
model= pm.auto_arima(Y_train, X_train, error_action="ignore")
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Non transformed: ", mean_absolute_error(Y_test, Y_pred))
print("r2_score Non transformed: ", r2_score(Y_test, Y_pred))

# build ARIMA model including vaccination predictions as exogenous variable
# using auto_arima on transformed data and print MAE and R-squared scores
model= pm.auto_arima(Y_train_trans, X_train)
Y_pred= model.predict(n_periods= len(Y_test), X= X_test)
print("mean_absolute_error Transformed: ", mean_absolute_error(Y_test_trans, Y_pred)**2)
print("r2_score Transformed: ", r2_score(Y_test_trans, Y_pred))

# write best performing model to a .sav file so it does not have to be run again in further use
icu_model_FRA= pm.auto_arima(Y_train_trans)
filename = 'models/icu_model_FRA.sav'
pickle.dump(icu_model_FRA, open(filename, 'wb'))
