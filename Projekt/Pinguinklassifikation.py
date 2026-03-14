import base64

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, ctx, State
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import RidgeClassifier
import joblib
import pandas as pd
import io

# Hier wird das Klassifikationsmodell verwaltet.
class Modell:

    # alle Variablen die innerhalb der Klasse genutzt werden, werden vorbereitet
    def __init__(self):
        self.penguins = pd.read_csv('penguins.csv')
        self.labels = pd.DataFrame()
        self.prep_penguins()
        self.modell = joblib.load("modell.pkl")
        self.penguins_scaled = self.scale_columns(self.vorbereitung(self.penguins))
        self.penguins_new = self.penguins_scaled

    # der initiale Dataframe mit den Palmer Pinguinen wird vorbereitet
    def prep_penguins(self):
        self.penguins.drop(columns='rowid', inplace=True)
        self.penguins.dropna(inplace=True)
        self.labels = pd.DataFrame({'labels': self.penguins['species'].copy()})
        self.penguins.drop(columns='species', inplace=True)
        # replace_dict = {'Torgersen': 0, 'Biscoe': 0.5, 'Dream': 1, 'male': 0, 'female': 1}
        self.vorbereitung(self.penguins)

    # Dataframes werden für die KI Modelle vorbereitet
    def vorbereitung(self, df):
        df.replace(to_replace='Torgersen', value=0, inplace=True)
        df.replace(to_replace='Biscoe', value=0.5, inplace=True)
        df.replace(to_replace='Dream', value=1, inplace=True)

        df.replace(to_replace='male', value=0, inplace=True)
        df.replace(to_replace='female', value=1, inplace=True)
        return df

    # Normalisierung
    def scale_columns(self, df):
        for col in df.columns:
            df[col] = minmax_scale(df[col])
            return df

    # Funkltion zum externen Aufruf der predict Methode
    def vorhersage(self, df):
        return self.modell.predict(df)

    # Modell aktualisieren, in dem der neue Dataframe zu den Trainingsdaten hinzugefügt wird.
    def modell_aktualisierung(self, erg, df):
        self.vorbereitung(df)
        self.penguins_new = pd.concat([self.penguins_new, df], ignore_index=True)
        lb = pd.DataFrame.from_dict({"labels": erg})
        self.labels = pd.concat([self.labels, lb], ignore_index=True)
        ridge = RidgeClassifier()
        self.modell = ridge.fit(self.penguins_new, self.labels['labels'])

    # bestehendes Modell abspeichern
    def modell_speichern(self, name):
        joblib.dump(self.modell, name)

    # neues Modell einstellen
    def modell_wechseln(self, name):
        self.modell = joblib.load(name)

#Hier sind alle global notwendigen Variablen
columns = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "year","Beobachter"]
og_penguins = pd.read_csv('penguins.csv')
m = Modell()

#Hier werden alle Icons für die Buttons erstellt
start_icon = DashIconify(icon="bi:play-circle", style={"marginRight": 5})
import_icon = DashIconify(icon="bi:file-earmark-arrow-up", style={"marginRight": 5})
delete_icon = DashIconify(icon="bi:trash", style={"marginRight": 5})
reset_icon = DashIconify(icon="bi:arrow-counterclockwise", style={"marginRight": 5})
download_icon = DashIconify(icon="bi:file-earmark-arrow-down", style={"marginRight": 5})
redo_icon = DashIconify(icon="bi:arrow-repeat", style={"marginRight": 5})
safe_icon = DashIconify(icon="bi:file-earmark-lock", style={"marginRight": 5})
switch_icon = DashIconify(icon="bi:arrow-left-right", style={"marginRight": 5})

#Hier wird das Layout der App erstellt und verwaltet
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
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
                               n_clicks=0, style={'textAlign': 'center'})),
            dbc.Col(dcc.Download(id="download-dataframe-csv"))
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
                        dbc.Input(id="input7", placeholder="Jahr (JJJJ)", type ='number'),
                        dbc.Input(id="input8", placeholder="Beobachter", type ='text'),
                        html.Br(),
                        html.H6(children='Ermittelte Spezies:', style={'textAlign': 'left'}),
                        html.Div(id='Ergebnis', style={'textAlign': 'center'},
                                 title= "Hier steht die vom Modell ermittelte Spezies.")
                        ]
        ),
            dbc.Col([dcc.Tabs(id='tabs-1', value='tab-1', style={'backgroundColor': '#fcfcfc'},
                     children=[dcc.Tab(label='Schnabelstruktur', value='tab-1'),dcc.Tab(label='Körperstruktur', value='tab-2'), dcc.Tab(label='Körpermaße', value='tab-3')]),
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
        ]),
        html.Br(),
        html.Div([dcc.Upload(id='upload', children=html.Div(['Upload per Drag & Drop oder hier eine ', html.A('Datei Auswählen')]),
                             style={
                                 'width': '100%',
                                 'height': '60px',
                                 'lineHeight': '60px',
                                 'borderWidth': '1px',
                                 'borderStyle': 'dashed',
                                 'borderRadius': '5px',
                                 'textAlign': 'center',
                                 'margin': '10px'})
                  ]),
        html.Div(id = 'file_in_upload', children=[''])
    ])
])

#hier werden alle Grafiken erstellt und die Auswahl über die Tabs verwaltet
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
            dcc.Graph(
                id='bill_structure',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig1
            )
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Graph(
                id='body_structure',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig2
            )
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.Graph(
                id='body_measurement',
                style={'backgroundColor': '#fcfcfc'},
                figure=fig3
            )
        ])

#hier wird die Funktionalität aller Buttons gesteuert. Jeder Button hat eine eigene ihm zugeordnete Funktion, wenn komplexere Vorgänge notwendig sind
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
    State('upload','filename'),
    State('upload', 'contents'),
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
def Click(btn1, btn2, btn3, btn5, btn6, btn7, btn8, filename, contents, erg, in1, in2, in3, in4, in5, in6, in7, in8):
    msg = " "
    if "berechnen" == ctx.triggered_id:
        arr = berechnung(in1, in2, in3, in4, in5, in6, in7, in8)
        erg = arr[1]
        msg = arr[0]
        return (msg, erg, in1,in2,in3,in4,in5,in6,in7,in8)
    elif "importieren" == ctx.triggered_id:
        arr = getCase(filename, contents)
        data_dict = arr[1]
        if data_dict == {}:
            (island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year,
             Beobachter) = ("Torgersen", "", "", "", "", "male", "", "")
            msg = arr[0]
        else:
            try:
                (island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year,
                 Beobachter) = (data_dict[key][0] for key in columns)
                msg = arr[0]
            except:
                (island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year,
                 Beobachter) =  ("Torgersen", "", "", "", "", "male", "", "")
                msg = "Die Datei ist nicht für den Import geeignet. Bitte wähle eine gültige Datei aus!"
        return (msg,"", island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year, Beobachter)
    elif "löschen" == ctx.triggered_id:
        return (msg, "", "Torgersen", "", "", "", "", "male", "", "")
    elif "speichern" == ctx.triggered_id:
        msg = safeCase(btn5)
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "aktualisieren" == ctx.triggered_id:
        msg = akt(erg, in1,in2,in3,in4,in5,in6,in7,in8)
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "modell_speichern" == ctx.triggered_id:
        msg = modell_speichern()
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    elif "modell_wechseln" == ctx.triggered_id:
        msg = modell_wechseln(filename)
        return (msg,erg,in1,in2,in3,in4,in5,in6,in7,in8)
    return (msg, erg, in1,in2,in3,in4,in5,in6,in7,in8)

#Diese Funktion verwaltet die Berechnung
def berechnung(in1, in2, in3, in4, in5, in6, in7, in8):
    val = [in1, in2, in3, in4, in5, in6, in7, in8]
    data = [val]
    ergebnis = ""
    try:
        eingabe=pd.DataFrame(data=data, columns=columns)
        eingabe.drop(columns='Beobachter', inplace=True)
        daten_scaled = m.scale_columns(m.vorbereitung(eingabe))
        try:
            ergebnis = m.vorhersage(daten_scaled)
            msg = "Die Berechnung wurde erfolgreich durchgeführt."
        except:
            msg = ("Das Modell konnte keine Vorhersage treffen. "
                    "Für Technische Probleme melde dich unter: katharina-maria.reichwein@iu-study.org")
    except:
        msg = "Bitte prüfe deine Eingaben!"
    return [msg, ergebnis]


#Diese Funktion verwaltet den Import der hochgeladenen Daten als CSV
def getCase(filename, contents):
    case = {}
    if not filename or not contents:
        msg = "Bitte lade zuerst eine .csv Datei hoch!"
    else:
        if "csv" in filename:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            case_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            case = case_df.to_dict()
            msg = "Die Daten wurden importiert."
        else:
            msg = "Die Datei konnte nicht geladen werden. Bitte wähle eine .csv Datei aus!"
    return [msg, case]


@callback(
    Output("file_in_upload", "children"),
    Input("upload", "filename")
)
def download(filename):
    msg = ""
    if not filename is None:
        msg = "Die Datei \"" + filename + "\" befindet sich im Upload"
    return msg

#Diese Funktion verwaltet das Speichern eines Falles als CSV
def safeCase(btn_5):
    msg = ""
    if btn_5 > 0:
        msg = "Die eingegebenen Daten wurden als neuer_Pinguin.csv gespeichert."
    else:
        msg = "Du hast den falschen Button betätigt."
    return msg

#hier wird die Antwort an das Download-Element aus der App verwaltet
@callback(
    Output("download-dataframe-csv", "data"),
    Input("speichern", "n_clicks"),
    State('Ergebnis', 'children'),
    State('input1', 'value'),
    State('input2', 'value'),
    State('input3', 'value'),
    State('input4', 'value'),
    State('input5', 'value'),
    State('input6', 'value'),
    State('input7', 'value'),
    State('input8', 'value'),
    prevent_initial_call=True
)
def download(n_clicks, erg, in1, in2, in3, in4, in5, in6, in7, in8):
    if erg[0] == "":
        vals = ("bisher wurde keine Vorhersage getroffen", in1, in2, in3, in4, in5, in6, in7, in8)
    else:
        vals = (erg[0], in1, in2, in3, in4, in5, in6, in7, in8)
    data = [vals]
    if n_clicks > 0:
        columns.insert(0, "species")
        data = pd.DataFrame(data=[[val for val in vals]], columns=columns)
        columns.pop(0)

        return dcc.send_data_frame(data.to_csv, "neuer_Pinguin.csv")



#Diese Funktion verwaltet das Aktualisieren des Modells
def akt(erg, *vals):
    try:
        data = [vals]
        daten = pd.DataFrame(data=data, columns=columns)
        daten.drop(columns='Beobachter', inplace=True)
        m.modell_aktualisierung(erg, daten)
        msg = "Das Modell wurde um den eingegebenen Pinguin erweitert."
    except:
        msg = ("Das Modell konnte nicht aktualisiert werden."
                        "Für Technische Probleme melde dich unter: katharina-maria.reichwein@iu-study.org")
    return msg

#Diese Funktion verwaltet das Speichern des Modells
def modell_speichern():
    try:
        m.modell_speichern("neues_Modell")
        return "Das Modell wurde gespeichert."
    except:
        return "Bitte wähle einen gültigen Speicherort aus!"


#Diese Funktion verwaltet den Wechsel eines Modells zu einem anderen
def modell_wechseln(filename):
    if not filename:
        return "Bitte wähle eine gültige Datei im .pkl Format aus!"
    else:
        try:
            m.modell_wechseln(filename)
            return "Das Modell wurde gewechselt."
        except:
            return "Bitte wähle eine gültige Datei im .pkl Format aus!"

if __name__ == '__main__':
    #Dashboard
    app.run(debug=True)

    #daten.drop(columns='Beobachter', inplace=True)
    #print(berechnung("Dream",27,10,57,5000,"male",2026,"Jorgo") )

    #Genauigkeitstest der verschiedenen Modelle testen
    #M = Modell()
    #X,y = M.penguins, M.labels
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    #knn=KNeighborsClassifier(algorithm='brute', n_neighbors=5)
    #knn.fit(X_train, y_train['labels'])
    #bernoulliNB = BernoulliNB()
    #bernoulliNB.fit(X_train, y_train['labels'])
    #ridge = RidgeClassifier()
    #ridge.fit(X, y['labels'])

    #def accuracy(modell):
    #    arr = modell.predict(X_test)
    #    korr = 0
    #    falsch = 0
    #    for i in range(len(y_test['labels'])):
    #        if arr[i] == y_test['labels'].iloc[i]:
    #            korr += 1
    #        else:
    #            falsch += 1
    #    return 'korrekt:' + str(korr) +  ' ,falsch:' + str(falsch)

    #print('Genauigkeit KNN')
    #print(accuracy(knn))

    #print('Genauigkeit Naive Bayes')
    #print(accuracy(bernoulliNB))

    #print('Genauigkeit Ridge')
    #print(accuracy(ridge))
