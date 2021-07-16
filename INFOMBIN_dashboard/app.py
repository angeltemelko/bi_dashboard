import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from data_preparation import USA_data, FRA_data, data
from prediction_plots import fig_ICU_USA, fig_ICU_FR, fig_cases_FR, fig_cases_USA, \
    fig_death_million_FR, fig_death_million_USA, fig_deaths_FR, fig_deaths_USA, fig_vacc_FR, fig_vacc_USA


USA = USA_data
FRA = FRA_data
FRA = FRA[:-2]

num_vacc = int(USA.iloc[-1, USA.columns.get_loc('total_vaccinations')])
all_cases_USA = int(USA.iloc[-1, USA.columns.get_loc('total_cases')])
all_vacc_USA = int(USA.iloc[-1, USA.columns.get_loc('people_vaccinated')])
all_deaths_USA = int(USA.iloc[-1, USA.columns.get_loc('total_deaths')])
new_cases_USA = int(USA.iloc[-1, USA.columns.get_loc('new_cases')])
new_deaths_USA = int(USA.iloc[-1, USA.columns.get_loc('new_deaths')])
pop_index_USA = USA.columns.get_loc("population")
pop_USA = USA.iloc[:, pop_index_USA]
pop_USA = int(pop_USA.tolist()[0])
perc_vacc_USA = round(all_vacc_USA / pop_USA * 100, 2)

all_cases_FRA = int(FRA.iloc[-1, FRA.columns.get_loc('total_cases')])
all_vacc_FRA = int(FRA.iloc[-1, FRA.columns.get_loc('people_vaccinated')])
all_deaths_FRA = int(FRA.iloc[-1, FRA.columns.get_loc('total_deaths')])
new_cases_FRA = int(FRA.iloc[-1, FRA.columns.get_loc('new_cases')])
new_deaths_FRA = int(FRA.iloc[-1, FRA.columns.get_loc('new_deaths')])
pop_index_FRA = FRA.columns.get_loc("population")
pop_FRA = FRA.iloc[:, pop_index_USA]
pop_FRA = int(pop_FRA.tolist()[0])
perc_vacc_FRA = round(all_vacc_FRA / pop_FRA * 100, 2)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('__name__', external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H3(children="Covid - 19",
                        style={
                            "margin-bottom": "0px",
                            "color": "white"}
                        ),
                html.H6(children="Visualize Covid-19 predictions generated from sources for USA and France.",
                        style={
                            "margin-top": "0px",
                            "color": "white"}
                        ),
            ])
        ], id="title"),
    ], id="header"),

    html.Div([
        html.Div([
            html.H6(children="Total Cases",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(id="cases",
                   style={
                       "textAlign": "center",
                       "color": "#ffa64d",
                       "fontSize": 30}
                   )], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Total Deaths",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(id="deaths",
                   style={
                       "textAlign": "center",
                       "color": "#0d1a80",
                       "fontSize": 30}
                   )], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Vaccinations",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(id="vacc",
                   style={
                       "textAlign": "center",
                       "color": "#79d279",
                       "fontSize": 30}
                   ),
            html.P(id="vacc-perc",
                   style={
                       "textAlign": "center",
                       "color": "#79d279",
                       "fontSize": 15,
                       "margin-top": "-18px"}
                   )
        ], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Population",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(id="pop",
                   style={
                       "textAlign": "center",
                       "color": "#ffff80",
                       "fontSize": 30}
                   )], className="card_container three columns"
        )
    ], className="row flex-display"),
    html.Div([
        html.Div([
            html.P("Select Country:", className="fix_label", style={"color": "white"}),
            dcc.Dropdown(id="w_countries",
                         multi=False,
                         clearable=True,
                         value="USA",
                         placeholder="Select Countries",
                         options=[{"label": c, "value": c}
                                  for c in ["USA", "FRA"]], className="dcc_compon"),
            html.Div([
                dcc.Graph(id="pie_chart",
                          config={"displayModeBar": "hover"}),
            ],),
        ], className="create_container six columns"),
        html.Div([
            html.P("Select:", className="fix_label", style={"color": "white"}),
            dcc.Dropdown(id="mode",
                         multi=False,
                         clearable=True,
                         value="People vaccinated per hundred",
                         placeholder="Select",
                         options=[{"label": c, "value": c}
                                  for c in ['People vaccinated per hundred', 'New cases per million', 'Total deaths', 'New deaths per million', 'ICU per million']],
                         className="dcc_compon"),
            dcc.Graph(id="line_chart")
        ], className="create_container six columns"),
    ], className="row flex-display"),
])


@app.callback(Output("pie_chart", "figure"),
              [Input("w_countries", "value")])
def update_graph(w_countries):
    covid_data = data.groupby(["date", "iso_code"])[["people_vaccinated", "people_fully_vaccinated", "population"]].sum().reset_index()
    covid_data = covid_data[covid_data["iso_code"] == w_countries]
    if w_countries == "FRA":
        covid_data = covid_data[:-2]
    fully_vaccinated = covid_data["people_fully_vaccinated"].iloc[-1]
    one_dose = covid_data["people_vaccinated"].iloc[-1] - fully_vaccinated
    no_vacc = covid_data["population"].iloc[-1] - one_dose - fully_vaccinated
    colors = ["#ffff80", "#40bf40", "#8cd98c"]
    return {
        "data": [go.Pie(labels=["Unvaccinated people", "People Fully Vaccinated", "People after one dose"],
                        values=[no_vacc, fully_vaccinated, one_dose],
                        marker=dict(colors=colors),
                        hoverinfo="label+value+percent",
                        textinfo="label+value",
                        textfont=dict(size=13),
                        hole=.7,
                        rotation=45)],
        "layout": go.Layout(
            plot_bgcolor="#2e6194",
            paper_bgcolor="#2e6194",
            hovermode="closest",
            title={
                "text": f"Vaccinations: {w_countries}",
                "y": 0.93,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"},
            titlefont={
                "color": "white",
                "size": 20},
            legend={
                "orientation": "h",
                "bgcolor": "#2e6194",
                "xanchor": "center", "x": 0.5, "y": -0.07},
            font=dict(
                family="sans-serif",
                size=12,
                color="white")
        ),
    }


@app.callback(
    Output('line_chart', 'figure'),
    [Input('w_countries', 'value'),
     Input('mode', 'value')])
def update_figure(w_countries, mode):
    if w_countries == 'FRA' and mode == 'People vaccinated per hundred':
        return fig_vacc_FR
    elif w_countries == 'USA' and mode == 'People vaccinated per hundred':
        return fig_vacc_USA
    elif w_countries == 'FRA' and mode == 'New cases per million':
        return fig_cases_FR
    elif w_countries == 'USA' and mode == 'New cases per million':
        return fig_cases_USA
    elif w_countries == 'FRA' and mode == 'Total deaths':
        return fig_deaths_FR
    elif w_countries == 'USA' and mode == 'Total deaths':
        return fig_deaths_USA
    elif w_countries == 'FRA' and mode == 'New deaths per million':
        return fig_death_million_FR
    elif w_countries == 'USA' and mode == 'New deaths per million':
        return fig_death_million_USA
    elif w_countries == 'FRA' and mode == 'ICU per million':
        return fig_ICU_FR
    elif w_countries == 'USA' and mode == 'ICU per million':
        return fig_ICU_USA


@app.callback(Output("cases", "children"),
              [Input("w_countries", "value")])
def update_cases(w_countries):
    if w_countries == 'USA':
        return f"{all_cases_USA:,}"
    elif w_countries == "FRA":
        return f"{all_cases_FRA:,}"


@app.callback(Output("deaths", "children"),
              [Input("w_countries", "value")])
def update_cases(w_countries):
    if w_countries == 'USA':
        return f"{all_deaths_USA:,}"
    elif w_countries == "FRA":
        return f"{all_deaths_FRA:,}"


@app.callback(Output("vacc", "children"),
              [Input("w_countries", "value")])
def update_cases(w_countries):
    if w_countries == 'USA':
        return f"{all_vacc_USA:,}"
    elif w_countries == "FRA":
        return f"{all_vacc_FRA:,}"


@app.callback(Output("vacc-perc", "children"),
              [Input("w_countries", "value")])
def update_cases(w_countries):
    if w_countries == 'USA':
        return f"{perc_vacc_USA}%"
    elif w_countries == "FRA":
        return f"{perc_vacc_FRA}%"


@app.callback(Output("pop", "children"),
              [Input("w_countries", "value")])
def update_cases(w_countries):
    if w_countries == 'USA':
        return f"{pop_USA:,}"
    elif w_countries == "FRA":
        return f"{pop_FRA:,}"


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
