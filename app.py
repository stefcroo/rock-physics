from email.quoprimime import body_check
from pickle import FALSE
from re import M
from this import d
from turtle import filling
import dash
import tk_tools
import tkinter as tk
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from scipy.interpolate import interp1d
import bruges 
from dash.exceptions import PreventUpdate
import math
import statistics
import time
from scipy.signal import hilbert
from dash.dash_table.Format import Format, Scheme, Trim
from flask_caching import Cache
import os
# Set environment variable
os.environ['TK_SILENCE_DEPRECATION'] = 1
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })


dictionary = {
                'Sandstone': {'lith':'Sandstone', 'lith_num':1, 'hatch': '..', 'color':'rgb(250, 248, 135,.2)'},
                'Shale': {'lith':'Shale', 'lith_num':1, 'hatch':'--', 'color':'rgb(191, 171, 140,.1)'},
                'Dolomite': {'lith':'Dolomite', 'lith_num':1, 'hatch':'-/', 'color':'rgb(185, 112, 250,.2)'},
                'Limestone': {'lith':'Limestone', 'lith_num':1, 'hatch':'+', 'color':'rgb(168, 240, 237,.2)'},
                'Anhydrite': {'lith':'Anhydrite', 'lith_num':1, 'hatch':'x', 'color':'rgb(202, 151, 247,.2)'},
                }

# @cache.memoize()
def castagna(lithology, depth_zbml):
    if lithology == 'Shale':
        vp = 5650-(4000)*math.exp(-2.4*10**-4*depth_zbml)
        vs = .7696*vp-.8673*1000
        rho  = 1.75*(vp/1000)**.265
    if lithology == 'Sandstone':
        vp = 5650- (4300)*math.exp(-3.8*10**-4*depth_zbml)
        vs =.8041*vp-.8558*1000
        rho  = 1.66*(vp/1000)**.261
    if lithology == 'Limestone':
        vp = 6800- (4300)*math.exp(-3.1*10**-4*depth_zbml)
        vs = 1000*(-.055*(vp/1000)*(vp/1000)+1.0167 *(vp/1000)-.990)
        rho  = 1.36*(vp/1000)**.386
    if lithology == 'Dolomite':
        vp = 6000- (4300)*math.exp(-3.1*10**-4*depth_zbml)
        vs = 1000*(.5832*(vp/1000)-.077)
        rho  = 1.74*(vp/1000)**.252
    if lithology == 'Anhydrite':
        vp = 6000- (4300)*math.exp(-3.1*10**-4*depth_zbml)
        vs = 1000*(.3532*(vp/1000)+.80741)
        rho  = 2.19*(vp/1000)**.160
    return (round(vp),round(vs),round(rho,2))


zbml_start = 1000
thickness = [30,20,30,20,30]
lithologies = ['Shale','Sandstone','Shale','Limestone','Anhydrite']
depths = [x+zbml_start for x in np.cumsum(thickness)]
top_depths = [x-y for x,y in zip(depths, thickness)]
vp_start,vs_start,rho_start=  ([] for i in range(3))

for i in range(len(lithologies)):
    vp, vs, rho = castagna(lithologies[i],depths[i])
    vs_start.append(vs)
    vp_start.append(vp)
    rho_start.append(rho)
start_df = pd.DataFrame(dict(THICKNESS = thickness, LITHOLOGY =lithologies, VP=vp_start, VS=vs_start, RHO=rho_start))

TWITTER_LOGO = 'twitterlogo.png'
GITHUB_LOGO = 'linkedin.jpeg'
LINKEDIN_LOGO = 'github.png'
MEDIUM_LOGO = 'medium.png'
TWITTER_HREF = 'https://twitter.com/StefCrooijmans'
GITHUB_HREF = 'https://github.com/stefcroo'
LINKEDIN_HREF = 'https://www.linkedin.com/in/stefan-crooijmans-71181095/'
MEDIUM_HREF = 'https://stefcroo.medium.com/'


navbar_style = {
    'backgroundColor': 'rgba(71, 71, 107)',
    'height': '200px',
    'color': 'rgba(255,255,255)',
}
# @cache.memoize()
def build_banner():
        return html.Div(
        id='banner',
        className="banner",
        children=[
            
            html.Div(
                id='app-page-header',
                children=[

                    html.H2(
                        'Forward Modeling Rock Physics Tool'
                    ),
                     html.A(
                        id='medium-logo', children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    base64.b64encode(
                                        open(
                                            '../assets/medium.png', 'rb'
                                        ).read()
                                    ).decode()
                                )
                            )],
                        href=MEDIUM_HREF
                    ),                                    
                                  
                    html.A(
                        id='twitter-logo', children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    base64.b64encode(
                                        open(
                                            '../assets/twitterlogo.png', 'rb'
                                        ).read()
                                    ).decode()
                                )
                            )],
                        href=TWITTER_HREF
                    ),    
                    html.A(
                        id='linkedin-logo', children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    base64.b64encode(
                                        open(
                                            '../assets/linkedin.jpeg', 'rb'
                                        ).read()
                                    ).decode()
                                )
                            )],
                        href=LINKEDIN_HREF
                    ),           
                    html.A(
                        id='github-logo', children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    base64.b64encode(
                                        open(
                                            '../assets/github.png', 'rb'
                                        ).read()
                                    ).decode()
                                )
                            )],
                        href=GITHUB_HREF
                    ),                

                ],
                style=navbar_style
                
            ),

        ]
    )

tab_style = {
    'borderBottom': '5px solid rgba(0,0,0)',
    'backgroundColor': 'rgba(148, 148, 255,.5)',
    'font-size': '1.2rem',
    'color': 'rgba(0,0,0)',
    'borderColor':' #23262E',
    'padding': '2px',
    'height': '60px',


    'display': 'flex',
    'flex-direction': 'column',
    'alignItems': 'center',
    'justifyContent': 'center'}

tab_selected_style =     {
    'borderBottom': '5px solid rgba(255,255,255)',
    'font-size': '1.2rem',

    'backgroundColor': 'rgba(148, 148, 255,.5)',
    'color': 'rgba(255,255,255)',
    'height': '60px',
    'padding': '2px',
    'fontWeight': 'bold',
    'display': 'flex',
    'flex-direction': 'column',
    'alignItems': 'center',
    'justifyContent': 'center'}
                        
slider_style  ={
        'margin-top': '20px',
        'margin-bottom': '5px'
}                  
col_style = {'margin':'4px',
        'backgroundColor': 'rgba(71, 71, 107)',
        'height': '600px'
         } 
col_style_data = {
    'margin':'4px',
    'backgroundColor': 'rgba(71, 71, 107)',

}

row1_style  ={
   'margin-top' : '150px',
   'margin-left' : '150px',
   'margin-right' : '150px'
            }

row2_style  ={
   'margin-top' : '20px',
   'margin-left' : '150px',
   'margin-right' : '150px'
}
app.title = "Rock Physics Forward Modeling"
app.layout = html.Div([
                build_banner(),

                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                    dcc.Tabs(id = 'intro-tabs', style ={'margin-top': '20px', 'margin-bottom': '20px'},  value ='Instructions', children  =[

                                        dcc.Tab(id = 'instructions', label = 'Instructions',  style=tab_style, selected_style=tab_selected_style, value = "Instructions", children = [
                                            html.H4(children='Create Your Subsurface Scenario'),
                                            html.Br(),
                                            html.Ul('1. Edit the input data in the table. Adjust the depth at the top of your model. Double click on the rock properties to edit them.',style= {'font-size': '1.1rem'}),
                                            html.Ul('2. Adjust the wavelet frequency and phase. ',style= {'font-size': '1.1rem'}),
                                            html.Ul('3. You have created a unique seismogram! Use the sliders to adjust the angle of incidence and scaling factors.', style= {'font-size': '1.1rem'}),
                                            html.Ul('4. Download the figures by hovering over the camera button.', style= {'font-size': '1.1rem'}),
                                            html.Ul('5. Share your work with colleagues, leave some feedback, and check out the source code using the links above!', style= {'font-size': '1.1rem'}),
                                        ]),
                                        dcc.Tab(id = 'about', label = 'About',  style=tab_style, selected_style=tab_selected_style, value = 'About', children = [
                                                html.H4(children='About Rock Physics'),
                                                html.Br(),
                                                html.P(["""
                                                Rock physics aims to relate geophysical measurements of rocks to geological processes.
                                                The seismic response of rocks is directly related to properties, including compressional wave velocity (Vp), shear wave velocity (Vs), and density (Rho).
                                                These values are dictated by the depositional history, provenance, and (chemical/mechanical) compaction trends that impact a rock's pore space, fluid fill, mineralogy, saturation, and clay-content.       
                                                """,
                                                html.Br(),html.Br(),

                                                """
                                                The acoustic properties can be estimated using empirical trends.
                                                The Input table contains pre-populated rock properties based on the Greenberg and Castagna equations for different lithologies. 

                                                """
                                    ]
                              )
                                        ]),
                                    ]),                
                ])
            ,  style=col_style )
                        
                
        
        
        
        ], width =3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Tabs(id = 'data-tabs', style ={'margin-top': '20px'}, value = 'Input Data', children = [
                                dcc.Tab(
                                    id = 'data-tab', value = 'Input Data', style=tab_style, selected_style=tab_selected_style, label = 'Input Data', children =  [
                                            html.Div([
                                                         dbc.Label('Top of Model', style = {'fontWeight': 'bold', 'font-size':'1.2rem', 'margin-left': '40px' }),
                                    dcc.Slider(id="zbml", tooltip = { 'always_visible': True},
                                    min=0, max=3000,step=10, value=1000, 
                                                                            marks = {       
                                                                            
                                                                                    0:{'label': '0m','style':{'color': 'rgba(255,255,255)', 'font-size':'1rem'}},
                                                                                    500: {'label': '500m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    1000:{'label': '1000m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    1500:{'label': '1500m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    2000:{'label': '2000m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    2500:{'label': '2500m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    3000:{'label': '3000m','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},                                                                                    
                                                                                    },
                                                                                    )], 
                                    style = slider_style
                                ),
                                html.Br(),html.Br(),
                                dbc.Label('Layered Model', style = {'fontWeight': 'bold', 'font-size':'1.2rem', 'margin-left': '40px' }),

                               
                                        
                                        dash_table.DataTable(id = 'our-table',
                                                data = start_df.to_dict('records'),

                                                columns=[                                                   
                                                        {'name': 'THICKNESS', 'id': 'THICKNESS', 'type':'numeric','deletable': False, 'renamable': False},
                                                        {'name': 'LITHOLOGY', 'id': 'LITHOLOGY', 'presentation':'dropdown'},
                                                        {'name': 'VP', 'id': 'VP','type':'numeric',  'deletable': False, 'renamable': False},
                                                        {'name': 'VS', 'id': 'VS', 'type':'numeric','deletable': False, 'renamable': False},
                                                        {'name': 'RHO', 'id': 'RHO','type':'numeric', 'deletable': False, 'renamable': False},
                                                        ],
                                                dropdown={                      #dictionary of keys that represent column IDs,
                                                        'LITHOLOGY': {                #its values are 'options' and 'clearable'
                                                            'options': [            #'options' represents all rows' data under that column
                                                                {'label': 'Sandstone', 'value': 'Sandstone'},
                                                                {'label': 'Shale', 'value': 'Shale'},
                                                                {'label': 'Limestone', 'value': 'Limestone'},
                                                                {'label': 'Dolomite', 'value': 'Dolomite'},
                                                                {'label': 'Anhydrite', 'value': 'Anhydrite'},
                                                            ],

                                                            'clearable':True
                                                        }},
                                                css=[{"selector": ".Select-menu-outer", "rule": "display: block !important", 'width': '100%'}],
                                                editable=True,
                                                row_deletable=True,             # allow user to delete rows
                                                sort_action="native",           # give user capability to sort columns
                                                sort_mode="single",             # sort across 'multi' or 'single' columns
                                                # filter_action="native",         # allow filtering of columns
                                                page_action='none',             # render all of the data at once. No paging.
                                                style_table={'overflowX': 'auto', 
                                                            'maxWidth': '2000px'},
                                                fixed_columns={'headers': True, 'data': 1},
                                                fill_width=True, 
                                                style_cell={'textAlign': 'center', 

                                                               'overflow': 'hidden',
                                                                'textOverflow': 'ellipsis',
                                                                'maxWidth': 0

                                                    },    

                                                style_cell_conditional=[ {
                                                                'backgroundColor': 'rgba(71, 71, 107)',
                                                                'color': 'rgba(255, 255, 255)',
                                                            },
                                                            {
                                                            'if': {
                                                                'column_id': 'LITHOLOGY'
                                                            },
                                                            'backgroundColor': 'rgb(240, 240, 240)',
                                                            'color': 'rgb(240, 240, 240)',

                                                           },
                                                  
                                                ],
                                                style_header={
                                                'backgroundColor': 'rgb(0, 0, 0)',
                                                'color': 'rgba(255, 255, 255)',
                                                'fontWeight': 'bold'
                                                 }
                                            ),
 
                                                            dbc.Button('Add Layer', id='add-row-button', outline = True, n_clicks = 0,  color = 'light', className ='me-1', style = {'margin-left': 10,'margin-top': 10,'width': 120}),
                                                            html.Div(id = 'output-text'),

                        ])
                                    
                                ,
                                dcc.Tab(
                                    id = 'output-tab', value = 'Derived Rock Properties', style=tab_style, selected_style=tab_selected_style, label = 'Derived Rock Properties', children =  [
                                            dcc.Loading(id='loading-2',   type = 'circle', children = [
                                                        html.Br(),

                                                        html.Div([html.Div(id = 'datatable-output')]),
                                                        html.Br(),
                                                        html.Div([html.Div(id = 'eei-datatable-output')]),

                                            ])
                                    ]
                                )]),
                            ]
                            )
                                                                                 
                    ], style =col_style_data)
                ],width = 5),
                dbc.Col([
                    dbc.CardBody([
                          dcc.Tabs(id = 'wavelet-tabs', style ={'margin-top': '20px'},  value ='Wavelet Frequency', children  =[
dcc.Tab(id = 'wavelet-frequency-tab', label = 'Wavelet Frequency', value= 'Wavelet Frequency',  
                                        style=tab_style, selected_style=tab_selected_style,
                                        children = [
                                html.Div([
                                    dcc.Slider(id="frequency", tooltip = { 'always_visible': True },
                                    min=20, max=100,step=1, value=60,  
                                                                            marks = {       
                                                                            
                                                                                    20: {'label': '20Hz','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    40: {'label': '40Hz','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    60: {'label': '60Hz','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    80: {'label': '80Hz','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                    100: {'label': '100Hz','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},},
                                                                                    )], 
                                    style = slider_style
                                )
                                    
                                ]),
                                dcc.Tab(id = 'phase-shift-tab', label = 'Phase Shift', 
                                style=tab_style, selected_style=tab_selected_style,                
                                children = [
                                html.Div(className = 'slider', children = [
                                        dcc.Slider(id="phase-shift", tooltip = { 'always_visible': True }, min=-180, max=180, step=5,value=0, 
                                                                    marks = {       
                                                                                        -180:{'label': '-180°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        -120:{'label': '-120°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        -60: {'label': '-60°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        0: {'label': '0°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        60: {'label': '60°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        120: {'label': '120°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        180: {'label': '180°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},
                                                                                        
                                                                                        }
                                       )

                                ], style = slider_style)
                                   

                                ]),
                            ]),
                            dcc.Loading(id='loading-wav',   type = 'circle', children = [
                                        
                                        html.Div([dcc.Graph(id='wavelet-fig')]),
                            ])

                    ], style= col_style)
                ], width =4)

                ], style = row1_style),
                dbc.Row([
                    dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            
                           dcc.Tabs(id='tabs-outputs', style ={'margin-top': '20px'}, children  =[
                                        dcc.Tab(
                                        id= 'max-angle-tab', 
                                        label = "Angle of Incidence", 
                                        style=tab_style, selected_style=tab_selected_style,
                                         children = [
                                    html.Div([


                                    dcc.Slider(id="max-angle", tooltip = { 'always_visible': True }, min=10, max=40, step=1,value=40, 
                                        marks = {       
                                          
                                                10: {'label':  '10°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                20: {'label':  '20°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                30: {'label':  '30°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                40: {'label':  '40°','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}
                                                }, 
                                            )
                                    ],style = slider_style)

                                ]
                                    ),

                                dcc.Tab(id = 'scale-tab', 
                                        label = 'Scaling Factor', 
                                         style=tab_style, selected_style=tab_selected_style,       
                                         children =[
                                    html.Div([
                                    dcc.Slider(id="scale", tooltip = { 'always_visible': True }, min=.5, max=50, step=.5,value=5, 
                                        marks = {       
                                            .5: {'label':  '.5','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                5: {'label':  '5','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                10: {'label':  '10','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}}, 
                                                20: {'label':  '20','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},                                                 
                                                30: {'label':  '30','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},                                                 
                                                40: {'label':  '40','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},                                                 
                                                50: {'label':  '50','style':{'color': 'rgba(255,255,255)','font-size':'1rem'}},                                                 
                                                
                                
                                                })
                                    ],style =slider_style
                                    )
                                    
                                ])
                            ]),
                            dcc.Loading(id='loading-synthetic',   type = 'circle', children = [
                                                        html.Div([dbc.Label('Synthetic Logs')], style = {'margin-bottom': '30px'}),
                                                        dcc.Graph(id='output-synthetic'),
                                            ])
                            ,
                            
                        ])
                    ], style =col_style)
                ], width =7)

                , 
                dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Tabs(id= 'output-fig-tabs', style ={'margin-top': '20px', 'margin-bottom':'40px'},children = [
                                    dcc.Tab(id ='avo-fig', label ='Amplitude vs Offset',  style=tab_style, selected_style=tab_selected_style,
children = [
                                        dcc.Loading(id='loading-avo',   type = 'circle', children = [
                                                        html.Div([ dcc.Graph(id='AvO-plot')]),
                                            ])
                                            
                                        ]),
                                        dcc.Tab(id ='eei-fig', label ='Extended Elastic Impedance', style=tab_style, selected_style=tab_selected_style,
children = [
                  dcc.Loading(id='loading-eei',   type = 'circle', children = [
                                                        html.Div([dcc.Graph(id='EEI-plot')]),
                                            ])

                                             
                                        ]),
                                        dcc.Tab(id ='ig-fig', label ='Intercept-Gradient', style=tab_style, selected_style=tab_selected_style,
children = [
                  dcc.Loading(id='loading-ig',   type = 'circle', children = [
                                                        html.Div([dcc.Graph(id='IG-plot')]),
                                            ])

                                             
                                        ]),
                                ]
                            
                                ),                              
                           
                            ])
                        , style =col_style)
                    ], width =5),
                ], style = row2_style),    


                  dbc.Row(html.Div(id='testing')) 
            ])

@app.callback(
    Output('our-table', 'data'),
    # Output('output-text', 'children'),

    [Input('add-row-button', 'n_clicks')],
    [State('zbml', 'value'),
    State('our-table', 'data'),
     State('our-table', 'columns')])
def add_row(n_clicks,zbml, rows, columns):
    # print(rows)
    df =pd.DataFrame(rows)
    df = df.replace('',0)
    layer_thickness = df['THICKNESS'].cumsum()
    
    z = layer_thickness[len(df)-1]+zbml
    a = len(rows)
    # Compute starting TWTf
    vp, vs, rho = castagna('Sandstone', z)
    if n_clicks > 0:
        rows.append(
        {'VP': vp, 'VS': vs, 'RHO': rho, 'THICKNESS': 30, 'LITHOLOGY': 'Sandstone'}
        )
    # b =rows[-2:]
    # return f'{b}'
    return rows

# @cache.memoize()
def ig_plot(df):
    intercepts =  []
    gradients = []
    labels = []
    fig = go.Figure()
    
    for i in range(len(df)-1):
        A = .5 * (((df['VP'].iloc[i+1]-df['VP'].iloc[i])/df.iloc[i]['VP']) + ((df['RHO'].iloc[i+1]-df['RHO'].iloc[i])/df.iloc[i]['RHO']))
        intercepts.append(A)
        B =  ((df['VP'].iloc[i+1]-df['VP'].iloc[i])/(2*df.iloc[i]['VP'])) - 4*((df['VP'].iloc[i]/df['VS'].iloc[i])**2)*((df['VS'].iloc[i+1]-df['VS'].iloc[i])/df.iloc[i]['VS']) - 2*((df['VP'].iloc[i]/df['VS'].iloc[i])**2)*((df['RHO'].iloc[i+1]-df['RHO'].iloc[i])/df.iloc[i]['RHO']) 
        gradients.append(B)
        text = f'Layer {df["LAYER"].iloc[i]} - {df["LAYER"].iloc[i+1]}'
        labels.append(text)
        fig.add_trace(go.Scatter(x= [A], y =[B],name = f'Layer {i+1}', marker_size=30, opacity =.8, showlegend=False))  
        fig.add_annotation(x= A+.15, y =B, showarrow=False, text = f'Layer {i} - Layer {i+1}', font=dict(size=16))

    df = pd.DataFrame(dict(A = intercepts, B = gradients, labels = labels))
    maximum = max(max(gradients),-min(gradients))


    # fig = go.Figure(data=go.Scatter(x=df['A'], y=df['B'], mode ='markers', marker = dict(size=12), showlegend=False))
    fig.update_xaxes(title_text="Intercept", range=[-1,1], zeroline=True,   zerolinewidth=2, zerolinecolor='white',gridcolor= 'rgba(204, 202, 202,.1)')
    fig.update_yaxes(title_text="Gradient (m)", range=[1.1*maximum,-1.1*maximum],  zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
    fig.update_layout(font_color="rgba(255, 255, 255, 1)",
    height = 550,
    title={
        'y':1,
        'x':0.5,
        'text': "Intercept vs Gradient at Interfaces",
        'xanchor': 'center',
        'yanchor': 'top'},
    )
   
    text1, text2, text3, text4="Class I","Class II","Class III","Class IV"
    
    return fig
# Add a callback to set he angle range
# @cache.memoize()
def rfc_plots(df, max_angle):
    theta_range = [x for x in range(0,max_angle,1)]
    
    fig  =go.Figure()
    vp, vs, rho = df['VP'], df['VS'],df['RHO']
    rfc_list = []
    for i in range(len(df)-1):
        rc = bruges.reflection.zoeppritz(vp.iloc[i], vs.iloc[i], rho.iloc[i], vp.iloc[i+1], vs.iloc[i+1], rho.iloc[i+1], theta_range)
        fig.add_trace(go.Scatter(x= theta_range, y =rc.real,mode = 'lines', line=dict(width=4), name = f'Layer {i}-{i+1}'))
        fig.add_annotation(x= 5, y =rc.real[0]+.01, showarrow=False, text = f'Layer {i+1}-{i+2}', font=dict(size=16))
        rfc_list.append(rc.real)
    maximum = max([max(sorted(l), key=abs) for l in rfc_list], key = abs)
    fig.update_xaxes(title_text="Angle θ", showline=True, linewidth=2, linecolor='black', zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    fig.update_yaxes(title_text="Reflection Coefficient", range=[-1*maximum-.1,maximum+.1],zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    fig.update_layout(font_color="rgba(255, 255, 255, 1)",hovermode="x unified", margin=dict(l=5, r=5, t=10, b=5), title={
        'y':1,
        'x':0.5,
        'text': "AvO Reflection Coefficients at Interfaces",
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})

    return fig 
# @cache.memoize()
def eei_calc(vp, vs, rho, chi,k):
    vp_norm = statistics.mean(vp)
    vs_norm = statistics.mean(vs)
    rho_norm = statistics.mean(rho)
    eei  =vp_norm*rho_norm*((vp/vp_norm)**(math.cos(math.radians(chi))+math.sin(math.radians(chi))))*((vs/vs_norm)**(-8*k*math.sin(math.radians(chi)))*(rho/rho_norm)**(math.cos(math.radians(chi))-4*k*math.sin(math.radians(chi))))
    return eei

# @cache.memoize()
def eei_fig(df):
    lith_fig = go.Figure()
    vp =df['VP']
    vs =df['VS']
    rho =df['RHO']

    vp_norm = statistics.mean(df['VP'])
    vs_norm = statistics.mean(df['VS'])
    rho_norm = statistics.mean(df['RHO'])
   
    chis = [x for x in range(-90,91)]
    eei_list = []
    k = .25
    eei_90,eei_60,eei_30,eei_0,eeimin30,eeimin60,eeimin90= eei_calc(vp,vs,rho,90,k),eei_calc(vp,vs,rho,60,k),eei_calc(vp,vs,rho,30,k),eei_calc(vp,vs,rho,0,k),eei_calc(vp,vs,rho,-30,k),eei_calc(vp,vs,rho,-60,k),eei_calc(vp,vs,rho,-90,k)
    eei_df = pd.DataFrame(dict(EEI90 = eei_90,EEI60 = eei_60,EEI30 = eei_30,EEI0 = eei_0,EEImin30 = eeimin30,EEImin60 = eeimin60,EEImin90 = eeimin90))
    fig = go.Figure()
    for i in range(len(df)):
        eei = [vp_norm*rho_norm*
        ((vp[i]/vp_norm)**(math.cos(math.radians(chi))+math.sin(math.radians(chi))))*
        ((vs[i]/vs_norm)**(-8*k*math.sin(math.radians(chi)))*
        (rho[i]/rho_norm)**(math.cos(math.radians(chi))-4*k*math.sin(math.radians(chi)))) for chi in chis]
        eei_list.append(eei)
        fig.add_trace(go.Scatter(x=chis, y =eei, line=dict(width=4), showlegend=False))
        fig.add_annotation(x= chis[15], y =eei[20], showarrow=False, text = f'Layer {i}', font=dict(size=16))
    maximum = max([max(sorted(l), key=abs) for l in eei_list], key = abs)+1000
    minimum = min([min(sorted(l), key=abs) for l in eei_list], key = abs)-1000
    fig.update_xaxes(title_text="Chi Angle", range= [-90,90],showline=True, linewidth=2, linecolor='black', zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    fig.update_yaxes(range=[minimum, maximum],zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    fig.update_layout(font_color="rgba(255, 255, 255, 1)", hovermode="x unified",margin=dict(l=5, r=5, t=10, b=5),
     title={
        'y':1,
        'x':0.5,
        'text': "Extended Elastic Impedance vs Chi Angle",
        'xanchor': 'center',
        'yanchor': 'top'},
        )
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})

    cols_df = [
    dict(id='EEI90', name='EEI90', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEI60', name='EEI60', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEI30', name='EEI30', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEI0', name='EEI0', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEImin30', name='EEI-30', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEImin60', name='EEI-60', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='EEImin90', name='EEI-90', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    ]

    eei_datatable = html.Div(
    dash_table.DataTable(id = 'eei-table',
                                                data =  eei_df.to_dict('records'),

                                                columns=cols_df,

                                                # css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
                   
                                                editable=False,                  # allow user to edit data inside tabel
                                                row_deletable=False,             # allow user to delete rows
                                                page_action='none',             # render all of the data at once. No paging.
                                                style_table={'overflowX': 'auto', 
                                                            'maxWidth': '2000px'},
                                                fixed_columns={'headers': True, 'data': 1},
                                                fill_width=True, 
                                                style_cell={'textAlign': 'center', 

                                                               'overflow': 'hidden',
                                                                'textOverflow': 'ellipsis',
                                                                'maxWidth': 0

                                                    },    
                                                    # css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],

                                                style_cell_conditional=[ {
                                                                'backgroundColor': 'rgba(71, 71, 107)',
                                                                'color': 'rgba(255, 255, 255)',
                                                            }
                                                  
                                                ],
                                                style_header={
                                                'backgroundColor': 'rgb(0, 0, 0)',
                                                'color': 'rgba(255, 255, 255)',
                                                'fontWeight': 'bold'
                                                 }
                                            ),
   )                                   
    return fig,eei_datatable
    
# def backus_ave(vp_list, vs_list, rho_list, backus, sample_rate_depth):
# # An effective-medium theory to derive seismic anisotropy of a long-wavelength equivalent medium from well-log data for synthetic seismogram manufacture. Involves harmonic averaging to find the anisotropic elastic parameters that characterize seismic-wave propagation at low frequencies in a layered medium. 
#     lb  = backus # Backus length
#     dz  = sample_rate_depth # Log sampling interval in meters 
#     # Bruges' backus takes numpy arrays in 
#     p = np.asarray(vp_list, dtype=np.float32)
#     s = np.asarray(vs_list, dtype=np.float32)
#     r = np.asarray(rho_list, dtype=np.float32)
#     vp_backus, vs_backus, rho_backus = bruges.rockphysics.backus(p, s, r, lb, dz)
#     return list(vp_backus), list(vs_backus), list(rho_backus)

# @cache.memoize()
def log_plots(df, max_angle, f, scale, phase):
    # depth_list = gen_depth_list(df)    
    depth_list,vp_list,vs_list,rho_list,ai_list,pr_list,AI_rfc_list,vpvs_list, lithology_list, lith_num,eei_list =  ([] for i in range(11))

    fig = make_subplots(rows=1, cols=7, 
                        shared_yaxes=True, 
                        horizontal_spacing = 0.00, 
                        # showlegend =False,
                        column_widths=[.12,.08,.08,.08,.08,.08, .48],
                        subplot_titles=("Lithology","Vp", "Vs", "Rho", "AI", "RFC", "AVO Synthetic"))

    vp_norm, vs_norm, rho_norm = statistics.mean(df['VP']), statistics.mean(df['VS']), statistics.mean(df['RHO'])
    angles = [0, max_angle,1]
    chis = [-90,-60, -45, -30, 0, 30, 45, 60, 90]
    k, sample_rate_depth = .25, .1524
    
    EEImin90, EEImin60, EEImin45, EEImin30, EEI0, EEIplus30, EEIplus45, EEIplus60, EEIplus90 = ([] for i in range(9))   
    eei_lists = [EEImin90, EEImin60, EEImin45, EEImin30, EEI0, EEIplus30, EEIplus45, EEIplus60, EEIplus90]
    for i in range(len(df)):
        # Create subsampled data for each layer
        depths = np.arange(df.iloc[i]['DEPTH'], df.iloc[i]['DEPTH']+df.iloc[i]['THICKNESS'], sample_rate_depth)
        depth_list+=list(depths)
        n = len(depths)

        # Add Lithology Data 
        lithology  = df.iloc[i]['LITHOLOGY']
        l_num = dictionary[lithology]['lith_num']
        l_nums = n*[l_num]
        l_liths = n*[lithology]
        color = dictionary[lithology]['color'] 
        fig.add_trace(go.Scatter(x=l_nums, y=  depths, fill ='tozerox', fillcolor= color, showlegend=False), row=1, col=1)
        fig.add_hline(y=depths[0], line= dict(width =4, color ='black'))        
        lithology = n*[lithology]
        lithology_list+=lithology
        eei = [vp_norm*rho_norm*
        ((df['VP'].iloc[i]/vp_norm)**(math.cos(math.radians(-90))+math.sin(math.radians(-90))))*
        ((df['VS'].iloc[i]/vs_norm)**(-8*k*math.sin(math.radians(chi)))*
        (df['RHO'].iloc[i]/rho_norm)**(math.cos(math.radians(-90))-4*k*math.sin(math.radians(-90)))) for chi in chis]

        vp = n*[df.iloc[i]['VP']]
        vs = n*[df.iloc[i]['VS']]
        rho = n*[df.iloc[i]['RHO']]
        ai = n*[df.iloc[i]['AI']]
        vpvs = n*[df.iloc[i]['VP/VS']]
        pr = n*[df.iloc[i]['PR']]
        vp_list+=vp
        vs_list+=vs
        rho_list+=rho
        ai_list+=ai
        vpvs_list+=pr
        pr_list+=pr

        for i in range(len(eei)):
            l = n*[eei[i]]
            eei_lists[i] +=l
    
    
    # reflects= []
    # RFCfig =go.Figure()
    # for i in range(len(vp_list)-1):
    #     refl = [bruges.reflection.zoeppritz(vp_list[i], vs_list[i],rho_list[i],
    #                                    vp_list[i+1], vs_list[i+1],rho_list[i+1], theta) for theta in angles]
    #     refl = [r.real for r in refl]
    #     reflects.append(refl)
    #     RFCfig.add_trace(go.Scatter(x= angles, y =refl))


    # vp_backus, vs_backus, rho_backus = backus_ave(vp_list, vs_list, rho_list, backus, sample_rate_depth)
    # fig.add_trace(go.Scatter(x= vp_backus, y = depth_list, marker_color='rgba(217, 86, 74, .9)'), row=1, col=2)
    # fig.add_trace(go.Scatter(x= vs_backus, y = depth_list, marker_color='rgba(245, 194, 66, .9)'), row=1, col=3)
    # fig.add_trace(go.Scatter(x= rho_backus, y = depth_list, marker_color='rgba(147, 242, 128,.9)'), row=1, col=4)
    EEIfig = go.Figure()
    for i in range(len(eei_list)):
        EEIfig.add_trace(go.Scatter(x= eei_list[i], y =depth_list,name = f'{eei_list[i]}'))

    # Reflection Coefficients in Depth at 0 
    for i in range(len(ai_list)-1):
        AI_rfc_list.append((ai_list[i+1]-ai_list[i])/(ai_list[i+1]+ai_list[i]))
    AI_rfc_list.append(AI_rfc_list[-1])    

    # Wavelet Parameters
    dt = .0001
    length = len(vp_list)*dt
    # t0 and y define the ricker wavelet
    phase_rad = phase*math.pi/180
    t0 = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t0**2)) * np.exp(-(np.pi**2)*(f**2)*(t0**2))
    x = hilbert(y)   
    y = math.cos(phase_rad)*x.real - math.sin(phase_rad)*x.imag


    wavelet_fig = go.Figure()
    wavelet_fig.add_trace(go.Scatter(x= t0, y =y, name='wavelet', line=dict(width=2, color = 'white')))
    wavelet_fig.update_yaxes(title_text="Amplitude", range=[min(y), max(y)], zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    wavelet_fig.update_xaxes(title_text="t0 (s)", range=[min(t0), max(t0)], zeroline=True, zerolinewidth=2, zerolinecolor='white',  gridcolor='rgba(204, 202, 202,.1)')
    wavelet_fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
    wavelet_fig.update_layout(font_color="rgba(255, 255, 255, 1)", height =435,title={
        'y':.9,
        'x':0.5,
        'text': "Wavelet",
        'xanchor': 'center',
        'yanchor': 'top'})

    max_time =[df['TWT'].iloc[-1]+ (2*df['THICKNESS'].iloc[-1]/df['VP'].iloc[-1])]
    # df['DEPTH'] = depth_list
    max_depth = [df['DEPTH'].iloc[-1] + df['THICKNESS'].iloc[-1]]
    ts = list(df['TWT'])+max_time
    zs =list(df['DEPTH'])+max_depth
    twt_f = interp1d(zs, ts)
    # Sub sample using steps
    twts = twt_f(depth_list)

    # Convolve the AI reflectivity series
    AI_synthetic = np.convolve(y, AI_rfc_list, mode='same')
    fig.add_trace(go.Scatter(x= AI_synthetic, y =depth_list ,marker_color='rgba(255, 255, 84, .9)'), row=1,col=6)
    fig.add_trace(go.Scatter(x= AI_rfc_list, y =depth_list,marker_color='rgba(255, 255, 84, .9)'), row=1,col=6)

    angles = range(0,max_angle,2)
    syn_lists = []
    syns = []

    for angle in angles:
        r_series =[]
        for i in range(len(vp_list)-1):
            refl = bruges.reflection.zoeppritz(vp_list[i], vs_list[i],rho_list[i],
                                    vp_list[i+1], vs_list[i+1],rho_list[i+1], angle) 
            r = refl.real                        
            r_series.append(r)
        r_series.append(r_series[-1])    
        angle_synthetic = np.convolve(y, r_series, mode='same')
        angle_synthetic = angle_synthetic *scale
        syns.append(angle_synthetic)
    
    for i in range(len(syns)):
        fig.add_trace(go.Scatter(x= [2*i + x for x in syns[i]], y =depth_list,mode='markers+lines',
                                    marker = {'size': 1},
                                    customdata =syns[i],
                                    
                                    hovertemplate='<b> %{customdata} </b>',
                                     line=dict(color='rgba(255, 255, 255, 1)', width=3), fill ='toself', fillcolor = 'rgba(255, 255, 255, .7)'), row=1,col=7)
    # Add traces to subplots
    fig.add_trace(go.Scatter(x= vp_list, y =depth_list,name = 'VP',line=dict(width=4), marker_color='rgba(240, 103, 146, .9)'), row=1,col=2)
    fig.add_trace(go.Scatter(x= vs_list, y =depth_list,name = 'VS' ,line=dict(width=4),marker_color='rgba(255, 126, 51, .9)'), row= 1, col=3)
    fig.add_trace(go.Scatter(x= rho_list, y =depth_list,name = 'RHO' ,line=dict(width=4),marker_color='rgba(213, 255, 5, .9)'), row=1,col=4)
    fig.add_trace(go.Scatter(x= ai_list, y =depth_list, name = 'AI' ,line=dict(width=4),marker_color='rgba(0, 255, 225, .9)'), row=1, col=5)
    # Update X axes
    fig.update_xaxes(title_font=dict(size=18), title_text="", range=[0,  .9],showticklabels=False,  mirror=True, showgrid =False, row =1, col=1)
    fig.update_xaxes(title_font=dict(size=18), range = [.9*min(vp_list), 1.1*max(vp_list)], title_text="m/s", showgrid=True, showline=True, mirror=True, linewidth=1, linecolor='white', gridcolor='rgba(204, 202, 202,.1)', row =1, col=2)
    fig.update_xaxes(title_text="m/s", range = [.9*min(vs_list), 1.1*max(vs_list)], showline=True, mirror=True, linewidth=1, linecolor='white', gridcolor='rgba(204, 202, 202,.1)', row =1, col=3)
    fig.update_xaxes(title_text="kg/m\u00b3", range = [.9*min(rho_list), 1.1*max(rho_list)], showline=True, mirror=True, linewidth=1, linecolor='white', gridcolor='rgba(204, 202, 202,.1)', row =1, col=4)
    fig.update_xaxes(title_text="g/s", range = [.9*min(ai_list), 1.1*max(ai_list)], showline=True, mirror=True, linewidth=1, linecolor='white', gridcolor='rgba(204, 202, 202,.1)', row =1, col=5)
    fig.update_xaxes(title_text="", showline=True, mirror=True, linewidth=1, linecolor='white', gridcolor='rgba(204, 202, 202,.1)' , row =1, col=6)
    fig.update_xaxes(title_text="", showgrid=True, showline=True, mirror=True, linewidth=1, linecolor='white',gridcolor='rgba(204, 202, 202,.1)',  row =1, col=7)   
    fig.update_xaxes(title_text="Angle of Incidence", range=[-1, max_angle], gridcolor='rgba(204, 202, 202,.1)', row=1, col=7)
    # Update Y axes
    fig.update_yaxes(title_text="Depth (m)", range = [max(zs), min(zs)], showgrid =False, row =1, col=1)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=2)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=3)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=4)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=5)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=6)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline=True, mirror=True, linewidth=1, linecolor='rgba(204, 202, 202,.2)', gridcolor='rgba(204, 202, 202,.1)', row =1, col=7)

    fig.update_layout(font_color="rgba(255, 255, 255, 1)", height = 360,hovermode='x unified', margin=dict(l=5, r=5, t=20, b=0), showlegend=False,
        title={
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},

        xaxis=dict(
        showline=True,
        showgrid=True,
        
        ))
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
    for i in range(len(df)):
        fig.add_annotation(x= .5, y =(df['DEPTH'].iloc[i]+df['THICKNESS'].iloc[i]/2), showarrow=False, font=dict(color = 'rgba(0,0,0)'), text = f'{df.iloc[i]["LITHOLOGY"]}')
    fig.add_annotation(x= .9*max(vp_list), y =.05*max(zs), row= 1, col=2, showarrow=False,text = 'VP', font=dict(
                                        size=15,
                                        color='rgba(240, 103, 146, .9)'))
    # fig.add_annotation(x= .9*max(vp_list), y =.1*max(zs), row= 1, col=2, showarrow=False,text = 'Backus', font=dict(
    #                                     size=15,
    #                                     color='rgba(217, 86, 74, .9)'))
    fig.add_annotation(x= .9*max(vs_list), y =.05*max(zs), row= 1, col=3, showarrow=False,text = 'VS', font=dict(
                                        size=15,
                                        color='rgba(255, 126, 51, .9)'))
    # fig.add_annotation(x= .9*max(vs_list), y =.1*max(zs), row= 1, col=3, showarrow=False,text = 'Backus', font=dict(
    #                                     size=15,
    #                                     color='rgba(245, 194, 66, .9)'))
    fig.add_annotation(x= .9*max(rho_list), y =.05*max(zs), row= 1, col=4, showarrow=False,text = 'Rho', font=dict(
                                        size=15,
                                        color='rgba(213, 255, 5, .9)'))
    # fig.add_annotation(x= .9*max(rho_list), y =.1*max(zs), row= 1, col=4, showarrow=False,text = 'Backus', font=dict(
    #                                     size=15,
    #                                     color='rgba(147, 242, 128,.9)'))
    return fig, wavelet_fig

@app.callback(
    [
    Output('output-synthetic', 'figure'),
    Output('AvO-plot', 'figure'), 
    Output('IG-plot', 'figure'),
    Output('EEI-plot', 'figure'), 
    Output("wavelet-fig", "figure"),
    Output('datatable-output', 'children'),
    Output('eei-datatable-output', 'children'),

    ]
    ,
    [Input('our-table', 'data'), Input('zbml', 'value'), Input('scale', 'value'), Input('frequency', 'value'), Input('max-angle','value'),  Input('phase-shift','value')])
def display_graph(data, zbml, scale, frequency, max_angle, phase):
    df =pd.DataFrame(data)
    df = df.replace('',0)

    df['LAYER']  = df.index+1
    df['AI'] =df['VP']* df['RHO']
    df['VP/VS'] =  df['VP']/df['VS']
    df['PR'] = ((df['VP']/df['VS'])**2-2)/(2*(df['VP']/df['VS'])**2-2) 
    df['K'] =  df['RHO']*((df['VP']*0.3048)**2-(4/3)*(df['VS']*0.3048)**2)*0.000001
    df['MU'] =  (df['VS']*0.3048)**2*df['RHO']*0.000001
    df['E'] =  9*df['K']*df['MU']/(3*df['K']+df['MU'])
    df['SI'] =  df['RHO']*df['VS']
    df['TOTAL_THICKNESS'] = df['THICKNESS'].cumsum()
    df['TWT_INTERVAL'] = 2*df['THICKNESS']/ df['VP']
    df['TOTAL_TWT'] = df['TWT_INTERVAL'].cumsum()

    # Compute starting TWT
    start_time = 2*zbml/1500
    depths = []
    times = []
    for i in range(len(df)):
        if i == 0:
            z = zbml
            time_ind = start_time
        else:
            z= df.iloc[i-1]['TOTAL_THICKNESS']+zbml
            time_ind  = df.iloc[i-1]['TOTAL_TWT']+start_time
        times.append(time_ind)
        depths.append(z)
    df['DEPTH'] = depths
    df['TWT'] = times 
    df_output =  df[['LAYER','DEPTH','TWT','AI','SI','VP/VS','PR','K','MU','E']]
    
    cols_dt = [
    dict(id='LAYER', name='LAYER', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='DEPTH', name='DEPTH', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='TWT', name='TWT', type='numeric', format=Format(precision=3, scheme=Scheme.fixed)),
    dict(id='AI', name='AI', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='SI', name='SI', type='numeric', format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='VP/VS', name='VP/VS', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='PR', name='POISSON RATIO', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='K', name='K', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='MU', name='MU', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='E', name='E', type='numeric', format=Format(precision=2, scheme=Scheme.fixed)),
    ]

    output_datatable = html.Div([
    dash_table.DataTable(id = 'derived-output-table',
                                                data =  df_output.to_dict('records'),
                                                columns=cols_dt,
                                                # css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
                                                editable=False,                  # allow user to edit data inside tabel
                                                row_deletable=False,             # allow user to delete rows
                                                page_action='none',             # render all of the data at once. No paging.
                                                style_table={'overflowX': 'auto', 
                                                            'maxWidth': '2000px'},
                                                fixed_columns={'headers': True, 'data': 1},
                                                fill_width=True, 
                                                style_cell={'textAlign': 'center', 

                                                               'overflow': 'hidden',
                                                                'textOverflow': 'ellipsis',
                                                                'maxWidth': 0

                                                    },    
                                                    # css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],

                                                style_cell_conditional=[ {
                                                                'backgroundColor': 'rgba(71, 71, 107)',
                                                                'color': 'rgba(255, 255, 255)',
                                                            },

                                                  
                                                ],
                                                style_header={
                                                'backgroundColor': 'rgb(0, 0, 0)',
                                                'color': 'rgba(255, 255, 255)',
                                                'fontWeight': 'bold'
                                                 }
                                            ),
    ])        
    IGplot = ig_plot(df)
    AVOplot  =rfc_plots(df, max_angle)
    EEIfig, eei_datatable = eei_fig(df)
    LOGplot, wavelet_fig = log_plots(df, max_angle,frequency, scale, phase)
    return LOGplot,AVOplot,IGplot, EEIfig,wavelet_fig, output_datatable,eei_datatable
    
if __name__=='__main__':
    app.run_server()