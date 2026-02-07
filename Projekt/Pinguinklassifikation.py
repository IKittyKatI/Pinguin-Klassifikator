import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, ctx, State
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from tkinter import filedialog
import csv
from dash_iconify import DashIconify
import Modell

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

columns = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "year","Beobachter"]
og_penguins = pd.read_csv('penguins.csv')
#user = pd.read_csv(r'D:\PyCharm\Übung\Benutzer.csv')
m = Modell.Modell()

start_icon = DashIconify(icon="bi:play-circle", style={"marginRight": 5})
import_icon = DashIconify(icon="bi:file-earmark-arrow-up", style={"marginRight": 5})
delete_icon = DashIconify(icon="bi:trash", style={"marginRight": 5})
reset_icon = DashIconify(icon="bi:arrow-counterclockwise", style={"marginRight": 5})
download_icon = DashIconify(icon="bi:file-earmark-arrow-down", style={"marginRight": 5})
redo_icon = DashIconify(icon="bi:arrow-repeat", style={"marginRight": 5})
safe_icon = DashIconify(icon="bi:file-earmark-lock", style={"marginRight": 5})
switch_icon = DashIconify(icon="bi:arrow-left-right", style={"marginRight": 5})

app.layout = dbc.Container([
    html.Div([
        html.H2(children='Pinguinklassifikator', style={'textAlign': 'center'}),
        html.Br(),
        dbc.Row([
            dbc.Col(dbc.Button(children= [start_icon, 'Start'], id='berechnen',
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dbc.Button(children= [import_icon,'Import'], id='importieren',
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dbc.Button(children= [delete_icon,'Löschen'], id='löschen',
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dbc.Button(children= [download_icon,'Speichern'], id='speichern',
                               n_clicks=0, style={'textAlign': 'center'}))
            ],
            justify="center"),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([dbc.Col(children=[
                        dcc.Dropdown(options =["Torgersen", "Biscoe", "Dream"],value="Torgersen", id="input1"),
                        dbc.Input(id="input2", placeholder="Schnabellänge in cm", type ='number'),
                        dbc.Input(id="input3", placeholder="Schnabeltiefe in cm", type ='number'),
                        dbc.Input(id="input4", placeholder="Flossenlänge in cm", type ='number'),
                        dbc.Input(id="input5", placeholder="Gewicht in g", type ='number'),
                        dcc.Dropdown(options =["male", "female"], value="male", id="input6"),
                        dbc.Input(id="input7", placeholder="Jahr (JJJJ)", type ='text'),
                        dbc.Input(id="input8", placeholder="Beobachter", type ='text'),
                        html.Br(),
                        html.H6(children='Ermittelte Spezies:', style={'textAlign': 'left'}),
                        html.Div(id='Ergebnis', style={'textAlign': 'center'},
                                 title= "Hier steht die vom Modell ermittelte Spezies.")
                        ]
        ),
            dbc.Col([dcc.Tabs(id='tabs-1', value='tab-1', style={'backgroundColor': '#fcfcfc'},
                     children=[dcc.Tab(label='Body', value='tab-1'),dcc.Tab(label='Body', value='tab-2'), dcc.Tab(label='Bill', value='tab-3')]),
                     html.Div(id='tabs-content')
                     ])
        ]),
        dbc.Row([dbc.Col(html.Div(id='Infos', style={'textAlign': 'center'}))]),
        html.Br(),
        html.H4(children='Modellanpassung:', style={'textAlign': 'left'}),
        dbc.Row([
            dbc.Col(dbc.Button(children=[redo_icon, 'Modell aktualisieren'], id='aktualisieren',
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dbc.Button(children=[safe_icon, 'Modell speichern'], id='modell_speichern',
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dbc.Button(children=[switch_icon, 'Modell wechseln'], id='modell_wechseln',
                               n_clicks=0, style={'textAlign': 'center'}))
        ])
    ])
])
@callback(
Output('tabs-content', 'children'),
Input('tabs-1', 'value'),
State('input1', 'value'),
State('input2', 'value'),
State('input3', 'value'),
State('input4', 'value'),
State('input5', 'value'),
State('input6', 'value'),
State('input7', 'value'),
Input('input8', 'value'))
def render(tab,in1,in2,in3,in4,in5,in6,in7,in8):
    if not in8:
        fig1 = px.scatter(og_penguins, x="bill_length_mm", y="bill_depth_mm",
                          color="species", symbol="island")
        fig2 = px.scatter(og_penguins, x="bill_length_mm", y="flipper_length_mm",
                          color="species", symbol="island")
        fig3 = px.scatter(og_penguins, x="bill_length_mm", y="body_mass_g",
                          color="species", symbol="island")
    else:
        fig1 = px.scatter(og_penguins, x="bill_length_mm", y="bill_depth_mm",
                          color="species", symbol="island").add_scatter(x=[in2], y=[in3],name='Pinguin')
        fig2 = px.scatter(og_penguins, x="bill_length_mm", y="flipper_length_mm",
                          color="species", symbol="island").add_scatter(x=[in2], y=[in4],name='Pinguin')
        fig3 = px.scatter(og_penguins, x="bill_length_mm", y="body_mass_g",
                          color="species", symbol="island").add_scatter(x=[in2], y=[in5],name='Pinguin')

    if tab == 'tab-1':
        return html.Div([
            html.Div(style={'backgroundColor': '#fcfcfc'}, children='body_structure'),
            dcc.Graph(
                id='body_structure',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig1
            )
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div(style={'backgroundColor': '#fcfcfc'}, children='bill_structure'),
            dcc.Graph(
                id='bill_structure',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig2
            )
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div(style={'backgroundColor': '#fcfcfc'}, children='bill_structure'),
            dcc.Graph(
                id='bill_structure',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig3
            )
        ])

@callback(
    Output('Infos', 'children'),
    Output('Ergebnis', 'children'),
    [Output('input1','value'),
    Output('input2', 'value'),
    Output('input3','value'),
    Output('input4','value'),
    Output('input5','value'),
    Output('input6','value'),
    Output('input7','value'),
    Output('input8','value')],

    Input('berechnen', 'n_clicks'),
    Input('importieren', 'n_clicks'),
    Input('löschen', 'n_clicks'),
    Input('speichern', 'n_clicks'),
    Input('aktualisieren', 'n_clicks'),
    Input('modell_speichern', 'n_clicks'),
    Input('modell_wechseln', 'n_clicks'),
    State('Ergebnis', 'children'),
    State('input1','value'),
    State('input2', 'value'),
    State('input3','value'),
    State('input4','value'),
    State('input5','value'),
    State('input6','value'),
    State('input7','value'),
    State('input8','value')
)
def Click(btn1, btn2, btn3, btn5, btn6, btn7, btn8, erg, in1, in2, in3, in4, in5, in6, in7, in8):
    msg = " "
    if "berechnen" == ctx.triggered_id:
        erg = berechnung(in1,in2,in3,in4,in5,in6,in7,in8)
        msg = "Die Berechnung wurde erfolgreich durchgeführt."
        return (msg, erg, in1,in2,in3,in4,in5,in6,in7,in8)
    elif "importieren" == ctx.triggered_id:
        data_dict = getCase()
        msg = "Die Daten wurden importiert"
        (island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year,
         Beobachter) = (data_dict[key][0] for key in columns)
        return (msg,"", island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year, Beobachter)
    elif "löschen" == ctx.triggered_id:
        return (msg, "", "Torgersen", "", "", "", "", "male", "", "")
    elif "speichern" == ctx.triggered_id:
        msg = safeCase(btn5, erg[0], in1,in2,in3,in4,in5,in6,in7,in8)
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "aktualisieren" == ctx.triggered_id:
        msg = akt(erg[0], in1,in2,in3,in4,in5,in6,in7,in8)
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "modell_speichern" == ctx.triggered_id:
        msg = modell_speichern()
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "modell_wechseln" == ctx.triggered_id:
        msg = modell_wechseln()
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    return (msg, erg, in1,in2,in3,in4,in5,in6,in7,in8)


def berechnung(*vals):
        try:
            daten=pd.DataFrame(data=[[val for val in vals]], columns=columns)
            daten.drop(columns='Beobachter', inplace=True)
            daten_scaled = m.scale_columns(m.vorbereitung(daten))
            ergebnis = m.vorhersage(daten_scaled)
            return ergebnis
        except Exception as e:
            return "Bitte prüfe deine Eingaben! Tipp: Führe zuerst die Berechnung durch."


def getCase():
    filepath = filedialog.askopenfilename(title="Wähle einen Einzelfall",
                                          filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    if not filepath:
        return
    else:
        try:
            case_df = pd.read_csv(filepath, sep=",",)
            case = case_df.to_dict()
            return case
        except:
            return

def safeCase(btn_5, *vals):
    msg = ""
    if btn_5 > 0:
        try:
            filepath = filedialog.asksaveasfilename(title="Wähle einen Speicherort",
                                                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                                                    initialfile="neuer_Pinguin.csv" )
            columns.insert(0,"species")
            data = [columns,  vals]
            with open(filepath, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(data)
            columns.pop(0)
            msg = "Die Berechnung wurde gespeichert."

        except:
            msg = "Die Berechnung konnte nicht gespeichert werden."

    else:
        msg = "Du hast den falschen Button betätigt."
    return msg

def akt(erg, *vals):
    try:
        daten = pd.DataFrame(data=[[val for val in vals]], columns=columns)
        daten.drop(columns='Beobachter', inplace=True)
        daten_scaled = m.scale_columns(m.vorbereitung(daten))
        m.modell_aktualisierung(erg, daten_scaled)
        msg = "Das Modell wurde um den eingegebenen Pinguin erweitert."
    except Exception as e:
        msg = "Bitte prüfe deine Eingaben!"
    return msg

def modell_speichern():
    filepath = filedialog.asksaveasfilename(title="Wähle einen Speicherort",
                                            filetypes=[("PKL Files", "*.pkl"), ("All Files", "*.*")],
                                            initialfile="neues_Modell.pkl")
    m.modell_speichern(filepath)
    return "Das Modell wurde gespeichert."

def modell_wechseln():
    filepath = filedialog.askopenfilename(title="Wähle ein Modell",
                                          filetypes=[("PKL Files", "*.pkl"), ("All Files", "*.*")])
    if not filepath:
        return "Bitte wähle eine gültige Datei aus!"
    else:
        m.modell_wechseln(filepath)
        return "Das Modell wurde gewechselt."

if __name__ == '__main__':
    app.run(debug=True)