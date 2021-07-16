# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:04:38 2021

@author: Benjamin
"""
import pandas as pd
import numpy as np
# =============================================================================
# import matplotlib.pyplot as plt
# =============================================================================
import pickle
import plotly.graph_objects as go
import statsmodels
from datetime import datetime

from data_preparation import USA_vaccination_data, USA_cases_data, USA_deaths_data, USA_icu_data, USA_R_data
from data_preparation import FRA_vaccination_data, FRA_cases_data, FRA_deaths_data, FRA_icu_data, FRA_R_data

# import seaborn as sns
# sns.set_style("whitegrid")

visual_horizon= [7, 31]

# =============================================================================
# USA and FRA vaccine prediction visualisation
# =============================================================================

# USA

data= USA_vaccination_data
n_data= len(data)

vaccine_model_USA = pickle.load(open('models/vaccine_model_USA.sav', 'rb'))

prediction_start, prediction_stop=n_data, n_data+ visual_horizon[1]
vaccine_predictions_USA= vaccine_model_USA.get_prediction(start= prediction_start, end= prediction_stop)

conf_int= vaccine_predictions_USA.conf_int()

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
dates= pd.date_range(start= current_date, periods= visual_horizon[1]+ 1).date
vaccine_predictions= vaccine_model_USA.predict(start= prediction_start, end= prediction_stop)

predicted_vaccination_USA= pd.DataFrame({'date':dates, 'people_fully_vaccinated_per_hundred':vaccine_predictions})

X_pred= predicted_vaccination_USA.iloc[:, 0]
Y_pred= predicted_vaccination_USA.iloc[:, 1]

full_date_range= pd.date_range(start= X_pred.iloc[0], end= X_pred.iloc[-1])


# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.fill_between(full_date_range,
#                  conf_int.iloc[:, 0], conf_int.iloc[:, 1],
#                  alpha=0.05, color='b')
# plt.ylim(0, 100)
# plt.xticks(rotation=45)
# plt.ylabel('Vaccination cover (in %)')
# plt.xlabel('date')
# plt.title("USA vaccination predictions")
# # plt.show()
# =============================================================================

# fig.savefig('plots/USA_vac_plot.png')

fig_vacc_USA = go.Figure()
fig_vacc_USA.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Vaccination cover (in %)'))
fig_vacc_USA.update_yaxes(range=[0, 100])

# France

data= FRA_vaccination_data
n_data= len(data)

vaccine_model_FRA= pickle.load(open('models/vaccine_model_FRA.sav', 'rb'))

prediction_start, prediction_stop=n_data, n_data+ visual_horizon[1]
vaccine_predictions_FRA= vaccine_model_FRA.get_prediction(start= prediction_start, end= prediction_stop)

conf_int= vaccine_predictions_FRA.conf_int()

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
dates= pd.date_range(start= current_date, periods= visual_horizon[1]+ 1).date
vaccine_predictions= vaccine_model_FRA.predict(start= prediction_start, end= prediction_stop)

predicted_vaccination_FRA= pd.DataFrame({'date':dates, 'people_fully_vaccinated_per_hundred':vaccine_predictions})

X_pred= predicted_vaccination_FRA.iloc[:, 0]
Y_pred= predicted_vaccination_FRA.iloc[:, 1]

full_date_range= pd.date_range(start= X_pred.iloc[0], end= X_pred.iloc[-1])

# =============================================================================
# fig= plt.figure()
# plt.fill_between(full_date_range,
#                  conf_int.iloc[:, 0], conf_int.iloc[:, 1],
#                  alpha=0.05, color='b')
# 
# #plt.plot(X_data, Y_data)
# plt.plot(X_pred, Y_pred)
# plt.ylim(0, 100)
# plt.xticks(rotation=45)
# plt.ylabel('Vaccination cover (in %)')
# plt.xlabel('date')
# plt.title("France vaccination predictions")
# # plt.show()
# =============================================================================

# fig.savefig('plots/FRA_vac_plot.png')

fig_vacc_FR = go.Figure()
fig_vacc_FR.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Vaccination cover (in %)'))
fig_vacc_FR.update_yaxes(range=[0, 100])

# =============================================================================
# Full vaccination dataset (from start to final prediction)
# =============================================================================
full_vac_USA= USA_vaccination_data.append(predicted_vaccination_USA)

full_vac_FRA= FRA_vaccination_data.append(predicted_vaccination_FRA)
# =============================================================================
# USA and FRA cases prediction visualisation
# =============================================================================

# USA
data= USA_cases_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

cases_model_USA = pickle.load(open('models/cases_model_USA.sav', 'rb'))

last_train_date= full_vac_USA.iloc[len(full_vac_USA)- horizon].iloc[0]
X_vac= full_vac_USA[full_vac_USA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = cases_model_USA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# plt.fill_between(full_date_range,
#                  conf_int[visual_horizon[1]-1:, 1], conf_int[visual_horizon[1]-1:, 0],
#                  alpha=0.05, color='b')
# =============================================================================

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Daily new cases')
# plt.xlabel('date')
# plt.title("USA daily new COVID-19 cases predictions")
# # plt.show()
# =============================================================================

# fig.savefig('plots/USA_cases_plot.png')

fig_cases_USA = go.Figure()
fig_cases_USA.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Daily new cases'))

# France

data= FRA_cases_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

cases_model_FRA = pickle.load(open('models/cases_model_FRA.sav', 'rb'))

last_train_date= full_vac_FRA.iloc[len(full_vac_FRA)- horizon].iloc[0]
X_vac= full_vac_FRA[full_vac_FRA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = cases_model_FRA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# plt.fill_between(full_date_range,
#                  conf_int[visual_horizon[1]-1:, 1], conf_int[visual_horizon[1]-1:, 0],
#                  alpha=0.05, color='b')
# =============================================================================

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Daily new cases')
# plt.xlabel('date')
# plt.title("FRA daily new COVID-19 cases predictions")
# # plt.show()
# =============================================================================

# fig.savefig('plots/FRA_cases_plot.png')

fig_cases_FR = go.Figure()
fig_cases_FR.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Daily new cases'))

# =============================================================================
# USA and FRA total deaths predictions visualisation
# =============================================================================

data= USA_deaths_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

total_deaths_model_USA = pickle.load(open('models/total_deaths_model_USA.sav', 'rb'))

last_train_date= full_vac_USA.iloc[len(full_vac_USA)- horizon].iloc[0]
X_vac= full_vac_USA[full_vac_USA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = total_deaths_model_USA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= predictions[-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Total deaths')
# plt.xlabel('Date')
# plt.title("USA total COVID-19 deaths predictions")
# # plt.show()
# # fig.savefig('plots/USA_deaths_plot.png')
# =============================================================================

fig_deaths_USA = go.Figure()
fig_deaths_USA.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Total deaths'))
# France

data= FRA_deaths_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

total_deaths_model_FRA = pickle.load(open('models/total_deaths_model_FRA.sav', 'rb'))

last_train_date= full_vac_FRA.iloc[len(full_vac_FRA)- horizon].iloc[0]
X_vac= full_vac_FRA[full_vac_FRA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = total_deaths_model_FRA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= predictions[-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Total deaths')
# plt.xlabel('Date')
# plt.title("France total COVID-19 deaths predictions")
# # plt.show()
# # fig.savefig('plots/FRA_deaths_plot.png')
# =============================================================================

fig_deaths_FR = go.Figure()
fig_deaths_FR.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Total deaths'))
# =============================================================================
# USA and FRA new daily deaths predictions visualisation
# =============================================================================

data= USA_deaths_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

new_deaths_model_USA = pickle.load(open('models/new_deaths_model_USA.sav', 'rb'))

last_train_date= full_vac_USA.iloc[len(full_vac_USA)- horizon].iloc[0]
X_vac= full_vac_USA[full_vac_USA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = new_deaths_model_USA.predict(n_periods= horizon, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('New deaths per million')
# plt.xlabel('Date')
# plt.title("USA daily COVID-19 deaths predictions")
# # plt.show()
# # fig.savefig('plots/USA_daily_deaths_plot.png')
# 
# =============================================================================
fig_death_million_USA = go.Figure()
fig_death_million_USA.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='New deaths per million'))
# France

data= FRA_deaths_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

new_deaths_model_FRA = pickle.load(open('models/new_deaths_model_FRA.sav', 'rb'))

last_train_date= full_vac_FRA.iloc[len(full_vac_FRA)- horizon].iloc[0]
X_vac= full_vac_FRA[full_vac_FRA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = new_deaths_model_FRA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('New deaths per million')
# plt.xlabel('Date')
# plt.title("France daily COVID-19 deaths predictions")
# # plt.show()
# # fig.savefig('plots/FRA_daily_deaths_plot.png')
# =============================================================================

fig_death_million_FR = go.Figure()
fig_death_million_FR.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='New deaths per million'))
# =============================================================================
# USA icu prediction visualisation
# =============================================================================

data= USA_icu_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

icu_model_USA = pickle.load(open('models/icu_model_USA.sav', 'rb'))

last_train_date= full_vac_USA.iloc[len(full_vac_USA)- horizon].iloc[0]
X_vac= full_vac_USA[full_vac_USA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = icu_model_USA.predict(n_periods= horizon, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= visual_horizon[1]).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Icu admissions (per million)')
# plt.xlabel('Date')
# plt.title("USA icu admissions (per million) predictions")
# # plt.show()
# # fig.savefig('plots/USA_icu_plot.png')
# =============================================================================

fig_ICU_USA = go.Figure()
fig_ICU_USA.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Icu admissions per million'))
# France

data= FRA_icu_data
n_data= len(data)
horizon= int(len(data)*0.2)+ visual_horizon[1]

icu_model_FRA = pickle.load(open('models/icu_model_FRA.sav', 'rb'))

last_train_date= full_vac_FRA.iloc[len(full_vac_FRA)- horizon].iloc[0]
X_vac= full_vac_FRA[full_vac_FRA['date']>= last_train_date]
X_vac= X_vac[['people_fully_vaccinated_per_hundred']]

predictions, conf_int = icu_model_FRA.predict(n_periods= horizon, X= X_vac, return_conf_int=True)

plot_data= data.iloc[n_data- visual_horizon[0]:]
X_data= plot_data.iloc[:, 0]
Y_data= plot_data.iloc[:, 1]

current_date= pd.to_datetime(str(datetime.now().date())).date()
X_pred= pd.date_range(start= current_date, periods= int(len(data)*0.2)-1).date
Y_pred= [value**2 for value in predictions][-visual_horizon[1]:]

full_date_range= pd.date_range(start= X_pred[0], end= X_pred[-1])

# =============================================================================
# fig= plt.figure()
# plt.plot(X_pred, Y_pred)
# plt.xticks(rotation=45)
# plt.ylabel('Icu admissions (per million)')
# plt.xlabel('Date')
# plt.title("France icu admissions (per million) predictions")
# # plt.show()
# # fig.savefig('plots/FRA_icu_plot.png')
# =============================================================================

fig_ICU_FR = go.Figure()
fig_ICU_FR.add_trace(go.Scatter(x=X_pred, y=Y_pred,
                    mode='lines',
                    name='Icu admissions per million'))
