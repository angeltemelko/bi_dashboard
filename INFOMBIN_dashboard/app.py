import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('__name__', external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
data = pd.read_csv("owid-covid-data.csv")

data = data[
    (data["iso_code"] == "NLD") | (data["iso_code"] == "USA")]
# data contains data from both the Netherlands and the USA.
data = data[["iso_code", "continent", "location", "total_cases", "new_cases", "total_deaths", "new_deaths",
             "new_deaths_smoothed", "total_cases_per_million", "new_cases_per_million",
             "new_cases_smoothed_per_million", "total_deaths_per_million", "new_deaths_per_million",
             "new_deaths_smoothed_per_million", "reproduction_rate", "icu_patients", "icu_patients_per_million",
             "hosp_patients", "hosp_patients_per_million", "weekly_icu_admissions", "weekly_icu_admissions_per_million",
             "weekly_hosp_admissions", "weekly_hosp_admissions_per_million", "new_tests", "total_tests",
             "total_tests_per_thousand", "new_tests_per_thousand", "new_tests_smoothed",
             "new_tests_smoothed_per_thousand", "positive_rate", "tests_per_case", "tests_units", "total_vaccinations",
             "people_vaccinated", "people_fully_vaccinated", "new_vaccinations", "new_vaccinations_smoothed",
             "total_vaccinations_per_hundred", "people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred",
             "new_vaccinations_smoothed_per_million", "stringency_index", "population", "population_density",
             "median_age", "aged_65_older", "aged_70_older", "gdp_per_capita", "extreme_poverty",
             "cardiovasc_death_rate", "diabetes_prevalence", "female_smokers", "male_smokers", "handwashing_facilities",
             "hospital_beds_per_thousand", "life_expectancy", "human_development_index", "excess_mortality"]]

# Data From the Netherlands only
# Entries start from 27/02/2020. 990 rows at time of observation.
Netherlands = data[data["iso_code"] == "NLD"]
Netherlands = Netherlands.drop_duplicates()
Netherlands = Netherlands.replace({"": np.nan, None: np.nan, "?": np.nan, 'NA': np.nan})

print(Netherlands)
print("Netherlands total deaths", Netherlands.iloc[-1, Netherlands.columns.get_loc('total_deaths')])
print("Netherlands total deaths per million",
      Netherlands.iloc[-1, Netherlands.columns.get_loc('total_deaths_per_million')])
print("Netherlands total number fully vaccinated",
      Netherlands.iloc[-1, Netherlands.columns.get_loc('people_fully_vaccinated')])

# Data from the USA Only
# entries start from 22/01/2020. 447 rows at time of observation.
USA = data[data["iso_code"] == "USA"]
USA = USA.drop_duplicates()
USA = USA.replace({"": np.nan, None: np.nan, "?": np.nan, 'NA': np.nan})

print(USA)
print("USA total deaths", USA.iloc[-1, USA.columns.get_loc('total_deaths')])
print("USA total deaths per million",
      USA.iloc[-1, USA.columns.get_loc('total_deaths_per_million')])
print("USA total number fully vaccinated",
      USA.iloc[-1, USA.columns.get_loc('people_fully_vaccinated')])

"""
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [14, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Stringency index dashboard'),

    html.Div(children='''
        This dashboard will illustrate how open a country will be.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)
"""
# print(covidData)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
