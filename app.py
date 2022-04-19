import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from scipy.interpolate import interp1d
import scipy
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
from whitenoise import WhiteNoise
import itertools

# Instantiate dash apprx
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY] ) 
# Define the underlying flask app (Used by gunicorn webserver in Heroku production deployment)
server = app.server 
# Enable Whitenoise for serving static files from Heroku (the /static folder is seen as root by Heroku) 
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/') 


dictionary = {
                'Sandstone': {'lith':'Sandstone', 'lith_num':1, 'hatch': '..', 'color':'rgb(250, 248, 135,.2)'},
                'Shale': {'lith':'Shale', 'lith_num':1, 'hatch':'--', 'color':'rgb(191, 171, 140,.1)'},
                'Dolomite': {'lith':'Dolomite', 'lith_num':1, 'hatch':'-/', 'color':'rgb(185, 112, 250,.2)'},
                'Limestone': {'lith':'Limestone', 'lith_num':1, 'hatch':'+', 'color':'rgb(168, 240, 237,.2)'},
                'Anhydrite': {'lith':'Anhydrite', 'lith_num':1, 'hatch':'x', 'color':'rgb(202, 151, 247,.2)'},
                }

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

def init_df():
    zbml_start, thickness, lithologies = 1000,[30,20,30,20,30], ['Shale','Sandstone','Shale','Limestone','Anhydrite']
    depths = [x+zbml_start for x in np.cumsum(thickness)]
    top_depths = [x-y for x,y in zip(depths, thickness)]
    vp_start,vs_start,rho_start= [x for i in range(len(lithologies)) for (x,y,z) in [castagna(lithologies[i], depths[i])]],[y for i in range(len(lithologies)) for (x,y,z) in [castagna(lithologies[i], depths[i])]],[z for i in range(len(lithologies)) for (x,y,z) in [castagna(lithologies[i], depths[i])]]
    start_df = pd.DataFrame(dict(THICKNESS = thickness, LITHOLOGY =lithologies, VP=vp_start, VS=vs_start, RHO=rho_start))
    return start_df

start_df = init_df()

TWITTER_LOGO = 'twitterlogo.png'
GITHUB_LOGO = 'linkedin.jpeg'
LINKEDIN_LOGO = 'github.png'
MEDIUM_LOGO = 'medium.png'
TWITTER_HREF = 'https://twitter.com/StefCrooijmans'
GITHUB_HREF = 'https://github.com/stefcroo/rock-physics'
LINKEDIN_HREF = 'https://www.linkedin.com/in/stefan-crooijmans-71181095/'
# MEDIUM_HREF = 'https://stefcroo.medium.com/'

navbar_style = {
    'backgroundColor': 'rgba(71, 71, 107)',
    'height': '60px',
    'color': 'rgba(255,255,255)',
    'padding-bottom': '100px',
    'display': 'inline-block',
    'vertical-align': 'top',
    'width': '100%'
}
subheader_style = {
    'font-size': '1.2rem',
    'color':'white',
    'margin-top': '10px'
}
waveletfig_style = {
    'width': '50%',
    'height': '80%'
}
logfig_style = {
    'margin-top': '10px', 
    'width': '90vh', 
    'height': '32vh'
}
avofig_style = {
    'margin-top': '20px', 
    'width': '65vh', 
    'height': '35vh'
}
eeifig_style = {
    'margin-top': '20px', 
    'width': '60vh', 
    'height': '35vh'
}
igfig_style = {
    'width': '65vh', 
    'height': '40vh'
}

# @cache.memoize()
def build_banner():
        return html.Div(
        id = 'banner',
        className="banner",
        style=navbar_style,

        children=[
                       html.Div(
                id = 'app-page-header',
                        style={
        
        },
                children=[
                html.Div([
                dbc.Row([
                    dbc.Col(children = [
                    html.H3(
                            'Rock Physics Forward Modeling', style={'margin-left':'10%'}
                    ),
                    ], width =9),
                    dbc.Col(
                        children = [
                    dbc.Button(
                        "Github",
                        id="github-link", 
                        href=GITHUB_HREF,
                        outline = True, n_clicks = 0,  color = 'light', className ='me-1', 
                        style = {'margin-left': '10px','margin-top': '0px','width': '30%'})
                        ,
                    dbc.Button(
                        "Linkedin",
                        id="linkedin-link", 
                        href=LINKEDIN_HREF,
                        outline = True, n_clicks = 0,  color = 'light', className ='me-1', style = {'margin-left': '10px','margin-top': '0px','width': '30%'}
                    )
                        ]
                    )
                ]
                )
                ], 
                style={'width': '100%', 'margin-top': '50px', 'display': 'inline-block', 'vertical-allign':'top'}
                )
                ]
                )
                ]
                )       
tab_style = {
    'borderBottom': '5px solid rgba(0,0,0)',
    'backgroundColor': 'rgba(148, 148, 255,.5)',
    'font-size': '1rem',
    'color': 'rgba(0,0,0)',
    'borderColor':' #23262E',
    'height': '35px',
    'display': 'flex',
    'flex-direction': 'column',
    'alignItems': 'center',
    'justifyContent': 'center'}

tab_selected_style = {
    'borderBottom': '5px solid rgba(255,255,255)',
    'font-size': '1rem',
    'backgroundColor': 'rgba(148, 148, 255,.5)',
    'color': 'rgba(255,255,255)',
    'height': '35px',
    'fontWeight': 'bold',
    'display': 'flex',
    'flex-direction': 'column',
    'alignItems': 'center',
    'justifyContent': 'center'}
                        
slider_style  = {
        'margin-top': '10px',
        'margin-bottom': '5px'
}                  
col_style = {'margin':'4px',
        'backgroundColor': 'rgba(71, 71, 107)',
        'height': '500px'
         } 
col_style_data = {
    'margin':'4px',
    'backgroundColor': 'rgba(71, 71, 107)',
}
row1_style  = {
   'margin-top' : '20px',
   'margin-left' : '30px',
   'margin-right' : '30px'
            }
row2_style  = {
   'margin-top' : '5px',
   'margin-left' : '30px',
   'margin-right' : '30px'
}
markdown_style = {
    'font-size': '.87rem',
    'color':'white',
    'margin-left': '5px'
}

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    [
        State('our-table', 'data'),
        State('zbml', 'value'),
    ],
    prevent_initial_call = True,
)
def func(n_clicks, rows,zbml):
    df =pd.DataFrame(rows)
    output_df = calculate_properties(df, zbml)
    return dcc.send_data_frame(output_df.to_csv, "InputRockPhysModel.csv")

@app.callback(
    Output('our-table', 'data'),
    [Input('add-row-button', 'n_clicks')],
    [State('zbml', 'value'),
    State('our-table', 'data'),
     State('our-table', 'columns')])
def add_row(n_clicks,zbml, rows, columns):
    # print(rows)
    df =pd.DataFrame(rows)
    if len(df)==0:
        PreventUpdate
    else:
        df = df.replace('',0)
        layer_thickness = df.THICKNESS.values.cumsum()
        z = layer_thickness[len(df)-1]+zbml
        a = len(rows)
        # Compute starting TWTf
        if n_clicks > 0:
            if len(df)%2==0 or len(df)==0:
                vp, vs, rho = castagna('Shale', z)
                rows.append(
                {'VP': vp, 'VS': vs, 'RHO': rho, 'THICKNESS': 30, 'LITHOLOGY': 'Shale'}
                )
            else:
                vp, vs, rho = castagna('Sandstone', z)
                rows.append(
                {'VP': vp, 'VS': vs, 'RHO': rho, 'THICKNESS': 20, 'LITHOLOGY': 'Sandstone'}
                )
        return rows

# @cache.memoize()
def ig_plot(df):
    if len(df) <= 1:
        PreventUpdate
    else:
        intercepts,gradients,labels =  ([] for i in range(3))
        fig = go.Figure()
        intercepts = [.5 * (((df.VP.values[i+1]-df.VP.values[i])/df.VP.values[i]) + ((df.RHO.values[i+1]-df.RHO.values[i])/df.RHO.values[i])) for i in range(len(df)-1)]
        gradients = [((df.VP.values[i+1]-df.VP.values[i])/(2*df.VP.values[i])) - 4*((df.VP.values[i]/df.VS.values[i])**2)*((df.VS.values[i+1]-df.VS.values[i])/df.VS.values[i]) - 2*((df.VP.values[i]/df.VS.values[i])**2)*((df.RHO.values[i+1]-df.RHO.values[i])/df.RHO.values[i]) for i in range(len(df)-1)] 
        maximum = max(max(gradients),-min(gradients))
        maxA = max(max(intercepts),-min(intercepts))
        for i in range(len(df)-1):
            label = f'{df.LITHOLOGY.values[i]} L{df.index[i]+1} -{df.LITHOLOGY.values[i+1]} L{df.index[i]+2}'
            fig.add_trace(go.Scatter(x= [intercepts[i]], y =[gradients[i]],name = label, marker_size=15, opacity =.8, showlegend = False))  
            fig.add_annotation(x = intercepts[i], y = gradients[i]+.05*maximum, showarrow = False, text = label, font = dict(size=12))
        fig.update_xaxes(title_text="Intercept", range=[-1.1*maxA,1.1*maxA], zeroline = True,   zerolinewidth = 2, zerolinecolor = 'white',gridcolor= 'rgba(204, 202, 202,.1)')
        fig.update_yaxes(title_text="Gradient (m)", range=[-1.1*maximum,1.1*maximum],  zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
        fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                                'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
        fig.update_layout(autosize = False,font_color="rgba(255, 255, 255, 1)",
        height = 420, width = 600
        )

        fig.add_trace(go.Scatter(
            x=[.1, 0, -.1, -.1],
            y=[-.7*maximum,-.7*maximum,-.7*maximum, .3*maximum],
            text=['Class I', 'Class II/IIp', 'Class III', 'Class IV'],
            textfont = dict(
            color="yellow"),
            showlegend = False,
            mode="text"
        ))        
        fig.update_shapes(dict(xref='paper', yref='paper'))

        return fig

def rfc_plots(df, max_angle):
    if len(df)<=1:
        PreventUpdate
    else:
        theta_range = [x for x in range(0,max_angle+1,1)]
        vp, vs, rho = df.VP.values, df.VS.values,df.RHO.values
        # Calculate Reflection Coefficients
        rc = [bruges.reflection.zoeppritz(vp[i], vs[i], rho[i], vp[i+1], vs[i+1], rho[i+1], theta_range) for i in range(len(df)-1)]
        rfc_list = [i.real for i in rc]
        # Create Figure
        fig  =go.Figure()
        for i in range(len(df)-1):
            fig.add_trace(go.Scatter(x= theta_range, y =rfc_list[i],mode = 'lines', line=dict(width = 2), name = f'L{i}-{i+1}'))
            fig.add_annotation(x= 15, y =rfc_list[i][0]+.01, showarrow=False, text = f'{df.LITHOLOGY.values[i]} L{i+1} - {df.LITHOLOGY.values[i+1]} L{i+2}', font = dict(size=12))
        # Set plot range based on data
        maximum = max([max(sorted(l), key=abs) for l in rfc_list], key = abs)
        fig.update_xaxes(title_text="Angle of Incidence", showline = True, range=  [0, max_angle], linewidth = 2, linecolor = 'black', zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
        fig.update_yaxes(title_text="Reflection Coefficient", range=[-1*maximum-.1,maximum+.1],zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
        fig.update_layout(font_color="rgba(255, 255, 255, 1)",hovermode="x unified", autosize = False, 
                        height = 400, width = 600,
                        margin=dict(l=5, r=5, t=5, b=5))
        fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                                'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
        return fig 

# Calculate EEI based on rock properties, chi angle
def eei_calc(vp, vs, rho, chi,k):
    vp_norm,vs_norm,rho_norm = statistics.mean(vp), statistics.mean(vs),statistics.mean(rho)
    eei  =vp_norm*rho_norm*((vp/vp_norm)**(math.cos(math.radians(chi))+math.sin(math.radians(chi))))*((vs/vs_norm)**(-8*k*math.sin(math.radians(chi)))*(rho/rho_norm)**(math.cos(math.radians(chi))-4*k*math.sin(math.radians(chi))))
    return eei

def eei_fig(df):
    vp,vs,rho =df.VP.values, df.VS.values,df.RHO.values
    vp_norm,vs_norm,rho_norm = statistics.mean(vp),statistics.mean(vs),statistics.mean(rho)











    chis,eei_list, k = [x for x in range(-90,91)],[], .25
    eei_90,eei_60,eei_30,eei_0,eeimin30,eeimin60,eeimin90= eei_calc(vp,vs,rho,90,k),eei_calc(vp,vs,rho,60,k),eei_calc(vp,vs,rho,30,k),eei_calc(vp,vs,rho,0,k),eei_calc(vp,vs,rho,-30,k),eei_calc(vp,vs,rho,-60,k),eei_calc(vp,vs,rho,-90,k)
    eei_df = pd.DataFrame(dict(EEI90 = eei_90,EEI60 = eei_60,EEI30 = eei_30,EEI0 = eei_0,EEImin30 = eeimin30,EEImin60 = eeimin60,EEImin90 = eeimin90))
    lith_fig = go.Figure()
    fig = go.Figure()
    # Loop through dataframe, calculate EEI at all chi angles and add trace to figure.
    for i in range(len(df)):
        eei = [vp_norm*rho_norm*
        ((vp[i]/vp_norm)**(math.cos(math.radians(chi))+math.sin(math.radians(chi))))*
        ((vs[i]/vs_norm)**(-8*k*math.sin(math.radians(chi)))*
        (rho[i]/rho_norm)**(math.cos(math.radians(chi))-4*k*math.sin(math.radians(chi)))) for chi in chis]
        eei_list.append(eei)
        fig.add_trace(go.Scatter(x=chis, y =eei, line=dict(width = 2), showlegend = False))
        fig.add_annotation(x= chis[25], y =eei[20], showarrow=False, text = f'{df.LITHOLOGY.values[i]} L{i}', font = dict(size=12,color= 'white'))
    
    # Set plot range based on data
    maximum = max([max(sorted(l), key=abs) for l in eei_list], key = abs)+1000
    minimum = min([min(sorted(l), key=abs) for l in eei_list], key = abs)-1000
    # Loop through dataframe, calculate EEI at all chi angles and add trace to figure.
    fig.update_xaxes(title_text="Chi Angle", range= [-90,90],showline = True, linewidth = 2, linecolor = 'black', zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
    fig.update_yaxes(range=[minimum, maximum],zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
    fig.update_layout(font_color="rgba(255, 255, 255, 1)", autosize = False,
                    height = 400, width = 600,
                    hovermode="x unified",margin=dict(l=5, r=5, t=5, b=5),
                        )
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})

    cols_df = [
    dict(id = 'EEI90', name = 'EEI90', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEI60', name = 'EEI60', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEI30', name = 'EEI30', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEI0', name = 'EEI0', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEImin30', name = 'EEI-30', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEImin60', name = 'EEI-60', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'EEImin90', name = 'EEI-90', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    ]

    eei_datatable = html.Div(
    dash_table.DataTable(id = 'eei-table',
                                                data =  eei_df.to_dict('records'),
                                                columns = cols_df,                   
                                                editable = False,                  # allow user to edit data inside tabel
                                                row_deletable = False,             # allow user to delete rows
                                                page_action = 'none',             # render all of the data at once. No paging.
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

def gen_wavelet(f, phase, depth_list):
    # Wavelet Parameters
    dt = .0001
    length = len(depth_list)*dt
    # t0 and y define the ricker wavelet
    phase_rad = phase*math.pi/180
    t0 = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t0**2)) * np.exp(-(np.pi**2)*(f**2)*(t0**2))
    # Apply phase shift
    x = hilbert(y)   
    y = math.cos(phase_rad)*x.real - math.sin(phase_rad)*x.imag
    return t0,x,y

def gen_wavelet_fig(t0, y):
    # Create figure for wavelet
    wavelet_fig = go.Figure()
    wavelet_fig.add_trace(go.Scatter(x= t0, y =y, name = 'wavelet', line=dict(width = 2, color = 'white')))
    wavelet_fig.update_yaxes(title_text="Amplitude", range=[min(y), max(y)], zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
    wavelet_fig.update_xaxes(title_text="t0 (s)", range=[min(t0), max(t0)], zeroline = True, zerolinewidth = 2, zerolinecolor = 'white',  gridcolor = 'rgba(204, 202, 202,.1)')
    wavelet_fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
    wavelet_fig.update_layout(autosize = False,font_color="rgba(255, 255, 255, 1)", 
        height = 320, width = 500    
    )
    return wavelet_fig

def log_plots(df, max_angle, f, scale, phase):
    # depth_list = gen_depth_list(df)    
    depth_list,vp_list,vs_list,rho_list,ai_list,pr_list,AI_rfc_list,vpvs_list, lithology_list, lith_num,eei_list =  ([] for i in range(11))
    fig = make_subplots(rows=1, cols=7, 
                        shared_yaxes=True, 
                        horizontal_spacing = 0.00, 
                        # showlegend =False,
                        column_widths=[.12,.08,.08,.08,.08,.08, .48],
                        subplot_titles=("Lithology","Vp", "Vs", "Rho", "AI", "RFC", "AVO Synthetic"))
    fig.update_annotations(font_size=12)

    angles, chis = [0, max_angle+1,1], range(-90,91,30)
    k, sample_rate_depth = .25, .1524
    EEImin90, EEImin60, EEImin45, EEImin30, EEI0, EEIplus30, EEIplus45, EEIplus60, EEIplus90 = ([] for i in range(9))   
    eei_lists = [EEImin90, EEImin60, EEImin45, EEImin30, EEI0, EEIplus30, EEIplus45, EEIplus60, EEIplus90]

    depths = [np.arange(df.DEPTH.values[i], df.DEPTH.values[i]+df.THICKNESS.values[i], sample_rate_depth) for i in range(len(df))]
    ns = [len(x) for x in depths]
    
    lithology = df.LITHOLOGY.values
    l_nums = [ns[i]*[1] for i in range(len(ns))]
    l_liths = [ns[i]*lithology[i] for i in range(len(ns))]
    color = [ns[i]*dictionary[df.LITHOLOGY.values[i]]['color'] for i in range(len(lithology))]
    vp,vs,rho, ai= [ns[i]*[df.VP.values[i]] for i in range(len(df))],[ns[i]*[df.VS.values[i]] for i in range(len(df))],[ns[i]*[df.RHO.values[i]] for i in range(len(df))],[ns[i]*[df.AI.values[i]] for i in range(len(df))]
    colors, depth_list, vp_list,vs_list,rho_list,ai_list= list(itertools.chain.from_iterable(color)),list(itertools.chain.from_iterable(depths)),list(itertools.chain.from_iterable(vp)),list(itertools.chain.from_iterable(vs)),list(itertools.chain.from_iterable(rho)),list(itertools.chain.from_iterable(ai))
    color = [dictionary[lithology[i]]['color']  for i in range(len(df))]

    for i in range(len(df)):
        fig.add_trace(go.Scatter(x=len(depths[i])*[1], y=  depths[i], fill ='tozerox', fillcolor= color[i], showlegend = False), row=1, col=1)

    # Reflection Coefficients in Depth at 0 
    
    AI_rfc_list=[(ai_list[i+1]-ai_list[i])/(ai_list[i+1]+ai_list[i]) for i in range(len(ai_list)-1)]
    AI_rfc_list.append(AI_rfc_list[-1])    
    t0,x,y = gen_wavelet(f, phase, depth_list)
    wavelet_fig = gen_wavelet_fig(t0,y)
    # Subsample time/depth by interpolating values to convert Reflectivity Series to time.  
    max_time =[df.TWT.values[-1]+ (2*df.THICKNESS.values[-1]/df.VP.values[-1])]
    # df['DEPTH'] = depth_list
    max_depth = [df.DEPTH.values[-1] + df.THICKNESS.values[-1]]
    ts = list(df.TWT.values)+max_time
    zs =list(df.DEPTH.values)+max_depth
    twt_f = interp1d(zs, ts)
    # Sub sample using steps
    twts = twt_f(depth_list)

    # Convolve the AI reflectivity series with derived wavelet
    AI_synthetic = np.convolve(y, AI_rfc_list, mode = 'same')
    
    # Create output figure with both reflectivity series and convolved trace
    fig.add_trace(go.Scatter(x= AI_synthetic, y =depth_list ,marker_color = 'rgba(255, 255, 84, .9)'), row=1,col=6)
    fig.add_trace(go.Scatter(x= AI_rfc_list, y =depth_list,marker_color = 'rgba(255, 255, 84, .9)'), row=1,col=6)
    maximumAI = max(AI_synthetic, key=abs)

    # Create AvO synhetic. Calculate every nth trace
    angles = range(0,max_angle+1,4)
    syn_lists, syns = ([] for i in range(2))

    for angle in angles:
        r_series =[(bruges.reflection.zoeppritz(vp_list[i], vs_list[i],rho_list[i],
                                    vp_list[i+1], vs_list[i+1],rho_list[i+1], angle)).real for i in range(len(vp_list)-1)]
        r_series.append(r_series[-1])    
        angle_synthetic = np.convolve(y, r_series, mode = 'same')*scale
        syns.append(angle_synthetic)
    # Add every trace to figure and add n*i to put the trace at right angle
    for i in range(len(syns)):
        fig.add_trace(go.Scatter(x= [4*i + x for x in syns[i]], y =depth_list,mode = 'lines',
                                    marker = {'size': 1},
                                     line=dict(color = 'rgba(255, 255, 255, 1)', width=1), fill ='toself', fillcolor = 'rgba(255, 255, 255, .7)'), row=1,col=7)
    # Add traces to subplots
    fig.add_trace(go.Scatter(x= vp_list, y =depth_list,name = 'VP',line=dict(width = 2), marker_color = 'rgba(240, 103, 146, .9)'), row=1,col=2)
    fig.add_trace(go.Scatter(x= vs_list, y =depth_list,name = 'VS' ,line=dict(width = 2),marker_color = 'rgba(255, 126, 51, .9)'), row= 1, col=3)
    fig.add_trace(go.Scatter(x= rho_list, y =depth_list,name = 'RHO' ,line=dict(width = 2),marker_color = 'rgba(213, 255, 5, .9)'), row=1,col=4)
    fig.add_trace(go.Scatter(x= ai_list, y =depth_list, name = 'AI' ,line=dict(width = 2),marker_color = 'rgba(0, 255, 225, .9)'), row=1, col=5)
    # Update X axes
    fig.update_xaxes(title_font = dict(size=14), title_text="", range=[0,  .9],showticklabels=False,  mirror=True, showgrid =False, row =1, col=1)
    fig.update_xaxes(title_font = dict(size=14), range = [.9*min(vp_list), 1.1*max(vp_list)], title_text="m/s", showticklabels=False, showgrid=True, showline = True, mirror=True, linewidth=1, linecolor = 'white', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=2)
    fig.update_xaxes(title_font = dict(size=14),title_text="m/s", range = [.9*min(vs_list), 1.1*max(vs_list)],showticklabels=False,showline = True, mirror=True, linewidth=1, linecolor = 'white', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=3)
    fig.update_xaxes(title_font = dict(size=14),title_text="g/cm\u00b3", range = [.9*min(rho_list), 1.1*max(rho_list)], showticklabels=False, showline = True, mirror=True, linewidth=1, linecolor = 'white', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=4)
    fig.update_xaxes(title_font = dict(size=14),title_text="kg/m\u00b3", range = [.9*min(ai_list), 1.1*max(ai_list)], showline = True, mirror=True, linewidth=1,  showticklabels=False, linecolor = 'white', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=5)
    fig.update_xaxes(title_font = dict(size=14),title_text="", range = [-1.1*maximumAI, 1.1*maximumAI], showticklabels=False, showline = True, mirror=True, linewidth=1, linecolor = 'white', gridcolor = 'rgba(204, 202, 202,.1)' , row =1, col=6)
    fig.update_xaxes(title_text="", showgrid=True, showline = True, mirror=True, ticktext=["0", "10", "20", "30", "40"], tickvals=[0, 10, 20, 30, 40], linewidth=1, linecolor = 'white',gridcolor = 'rgba(204, 202, 202,.1)',  row =1, col=7)   
    fig.update_xaxes(title_text="Angle of Incidence", range=[-3, max_angle+5], gridcolor = 'rgba(204, 202, 202,.1)', row=1, col=7)
    # Update Y axes
    fig.update_yaxes(title_text="Depth (m)", range = [max(zs), min(zs)], showgrid =False, row =1, col=1)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=2)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=3)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=4)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=5)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=6)
    fig.update_yaxes(title_text="", range = [max(zs), min(zs)],showline = True, mirror=True, linewidth=1, linecolor = 'rgba(204, 202, 202,.2)', gridcolor = 'rgba(204, 202, 202,.1)', row =1, col=7)

    fig.update_layout(font_color="rgba(255, 255, 255, 1)", 
    # height = 360,
                    hovermode = 'x unified', margin=dict(l=5, r=5, t=15, b=5), 
                    autosize = False,
                    height = 320, width = 900,
                    showlegend = False,
                        xaxis=dict(
                        showline = True,
                        showgrid=True,
                        
                        ))
    fig.update_layout({'plot_bgcolor': 'rgba(43, 43, 64,1)',
                            'paper_bgcolor': 'rgba(71, 71, 107, 1)'})
    # Add annotation to plot to indicate layers
    for i in range(len(df)):
        fig.add_annotation(x= .5, y =(df.DEPTH.values[i]+df.THICKNESS.values[i]/2), showarrow=False, font = dict(color = 'rgba(0,0,0)'), text = f'{df.LITHOLOGY.values[i]}')
    fig.add_annotation(x= .9*max(vp_list), y =.05*max(zs), row= 1, col=2, showarrow=False,text = 'VP', font = dict(
                                        size=15,
                                        color = 'rgba(240, 103, 146, .9)'))

    fig.add_annotation(x= .9*max(vs_list), y =.05*max(zs), row= 1, col=3, showarrow=False,text = 'VS', font = dict(
                                        size=15,
                                        color = 'rgba(255, 126, 51, .9)'))
   
    fig.add_annotation(x= .9*max(rho_list), y =.05*max(zs), row= 1, col=4, showarrow=False,text = 'Rho', font = dict(
                                        size=15,
                                        color = 'rgba(213, 255, 5, .9)'))
    return fig, wavelet_fig

def calculate_properties(df, zbml):
    layers = df.index+1
    total_thickness = df.THICKNESS.values.cumsum()
    twt_interval =2*df.THICKNESS.values/ df.VP.values
    total_twt = twt_interval.cumsum()
    # Compute depth and TWT at top of Layer
    start_time = 2*zbml/1500
    depths,times = ([] for i in range(2))
    depths = [zbml if i==0 else total_thickness[i-1]+zbml for i in range(len(df))]
    times= [start_time if i==0 else total_twt[i-1]+start_time for i in range(len(df))]
    ai = df.VP.values* df.RHO.values
    vpvs = df.VP.values/df.VS.values
    pr = ((df.VP.values/df.VS.values)**2-2)/(2*(df.VP.values/df.VS.values)**2-2)
    k = df.RHO.values*((df.VP.values*0.3048)**2-(4/3)*(df.VS.values*0.3048)**2)*0.000001
    mu = (df.VS.values*0.3048)**2*df.RHO.values*0.000001
    e =  9*k*mu/(3*k+mu)
    si  = df.RHO.values*df.VS.values

    df['DEPTH'], df['TWT'],df['AI'] = depths, times, ai

    df_output =  pd.DataFrame(dict(LAYER = layers, DEPTH =depths, TWT=times, AI=ai, VP_VS=vpvs, PR=pr, K=k, MU=mu, E=e, SI=si))
    cols_dt = [
    dict(id = 'LAYER', name = 'LAYER', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'DEPTH', name = 'DEPTH', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'TWT', name = 'TWT', type = 'numeric', format = Format(precision=3, scheme = Scheme.fixed)),
    dict(id = 'AI', name = 'AI', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'SI', name = 'SI', type = 'numeric', format = Format(precision=0, scheme = Scheme.fixed)),
    dict(id = 'VP/VS', name = 'VP/VS', type = 'numeric', format = Format(precision=2, scheme = Scheme.fixed)),
    dict(id = 'PR', name = 'POISSON RATIO', type = 'numeric', format = Format(precision=2, scheme = Scheme.fixed)),
    dict(id = 'K', name = 'K', type = 'numeric', format = Format(precision=2, scheme = Scheme.fixed)),
    dict(id = 'MU', name = 'MU', type = 'numeric', format = Format(precision=2, scheme = Scheme.fixed)),
    dict(id = 'E', name = 'E', type = 'numeric', format = Format(precision=2, scheme = Scheme.fixed)),
    ]
    # Create output datatable

    output_datatable = html.Div([
        dash_table.DataTable(id = 'derived-output-table',
                                                    data =  df_output.to_dict('records'),
                                                    columns = cols_dt,
                                                    editable = False,                  # allow user to edit data inside tabel
                                                    row_deletable = False,             # allow user to delete rows
                                                    page_action = 'none',             # render all of the data at once. No paging.
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
    return df,output_datatable

@app.callback(
    [
    Output('output-synthetic', 'figure'),
    Output('AvO-plot', 'figure'), 
    Output('IG-plot', 'figure'),
    Output('EEI-plot', 'figure'), 
    Output('datatable-output', 'children'),
    Output('eei-datatable-output', 'children'),
    Output('wavelet-fig', 'figure')
    ]
    ,
    [Input('our-table', 'data'), Input('zbml', 'value'), Input('scale', 'value'), Input('frequency', 'value'), Input('max-angle','value'),  Input('phase-shift','value')])
def display_graph(data, zbml, scale, frequency, max_angle, phase):
    df =pd.DataFrame(data)
    if len(df)==0:
        PreventUpdate
    else:
        df = df.replace('',0)
        df, output_datatable = calculate_properties(df, zbml)
        IGplot = ig_plot(df)
        AVOplot  =rfc_plots(df, max_angle)
        EEIfig, eei_datatable = eei_fig(df)
        LOGplot,wavelet_fig = log_plots(df, max_angle,frequency, scale, phase)
        return LOGplot,AVOplot,IGplot, EEIfig, output_datatable,eei_datatable,wavelet_fig
    
def create_dash_layout(app):
    # Set browser tab title
    app.title = "Rock Physics Forward Modeling"
        
    app.layout = html.Div(style = {'backgroundColor':'rgba(43, 43, 64,1)'}, children = [
                    build_banner(),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                        dcc.Tabs(id = 'intro-tabs', style ={'margin-top': '10px', 'margin-bottom': '3px'},  value ='Instructions', children  =[

                                            dcc.Tab(id = 'instructions', label = 'Instructions',  style=tab_style, selected_style=tab_selected_style, value = "Instructions", children = [
                                                html.Div([
                                                    html.H4(children = 'Create A Subsurface Scenario'),
                                                

                                                ], style=subheader_style),
                                                html.Br(),
                                                html.Ul('1. Edit the input parameters. Adjust the depth at the top of your layered model and edit the input data table by clicking on the cells and editing the number of layers.',style= markdown_style),
                                                html.Ul('2. Adjust the frequency and phase of the wavelet used to convolve the reflectivity series.',style= markdown_style),
                                                html.Ul('3. Analyse the results. Customize the AvO synthetic using the sliders.', style= markdown_style),
                                                html.Ul('4. Download your custom data table and results by hovering over the figures.', style= markdown_style),
                                                html.Ul('5. Share your work with colleagues, leave some feedback, and check out the source code!', style= markdown_style),

                                            ]),
                                            dcc.Tab(id = 'about', label = 'About',  style=tab_style, selected_style=tab_selected_style, value = 'About', children = [
                                                    html.Div([
                                                        html.H4(children = 'About Rock Physics')

                                                    ], style=subheader_style),
                                                    html.P(["""
                                                    The amplitude response of rocks measured by seismic data is directly related to contrasts in rock properties like compressional wave velocity (Vp), shear wave velocity (Vs), and density (Rho) across bounding lithologies.
                                                    These rock properties are impacted by basin scale factors like provenance, diagenesis, and mechanical compaction which affect the pore space, fluid fill, mineralogy, saturation, and clay-content in rock samples. 
                                                    """,
                                                    html.Br(),html.Br(),
                                                    """
                                                    Amplitude response as a function of offset (AvO) and Extended Elastic Impedance are commonly used to discriminate lithology and fluid response and can be derived from Vp, Vs, and Rho. 
                                                    """,

                                                    html.Br(), html.Br(),

                                                    """
                                                    Rock properties can be estimated using empirical trends for different lithologies.
                                                    The Input data table (right) contains pre-populated rock properties based on the Greenberg and Castagna equations.   


                                                    """
                                        ], style = markdown_style
                                )
                                            ]),
                                        ]),                
                    ])
                ,  style=col_style )
                            
                    
            
            
            
            ], width =3),
                    
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Tabs(id = 'data-tabs', style ={'margin-top': '10px','margin-bottom': '3px'}, value = 'Input Data', children = [
                                        dcc.Tab(
                                            id = 'data-tab', value = 'Input Data', style=tab_style, selected_style=tab_selected_style, label = 'Input Data', children =  [
                                                    html.Div([
                                                                dbc.Label('Top of Model', style = {'fontWeight': 'bold', 'font-size':'1rem', 'margin-left': '40px' , 'color':'white'}),
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
                                        html.Br(),
                                        dbc.Label('Layered Model', style = {'fontWeight': 'bold', 'font-size':'1rem', 'margin-left': '40px','color':'white' }),
                                                dash_table.DataTable(id = 'our-table',
                                                        data = start_df.to_dict('records'),
                                                        columns=[                                                   
                                                                {'name': 'THICKNESS (m)', 'id': 'THICKNESS', 'type':'numeric','deletable': False, 'renamable': False},
                                                                {'name': 'LITHOLOGY', 'id': 'LITHOLOGY', 'presentation':'dropdown'},
                                                                {'name': 'VP (m/s)', 'id': 'VP','type':'numeric',  'deletable': False, 'renamable': False},
                                                                {'name': 'VS (m/s)', 'id': 'VS', 'type':'numeric','deletable': False, 'renamable': False},
                                                                {'name': 'RHO (g/cm\u00b3)', 'id': 'RHO','type':'numeric', 'deletable': False, 'renamable': False},
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
                                                        page_action = 'none',             # render all of the data at once. No paging.
                                                        style_table={'overflowX': 'auto', 
                                                                    'maxWidth': '2000px'},
                                                        fixed_columns={'headers': True, 'data': 1},
                                                        fill_width=True, 
                                                        style_cell={'textAlign': 'center', 
                                                                    'fontSize':14,
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
                                                    dbc.Row([
                                                        dbc.Col([
                                                            dbc.Button('Add Layer', id = 'add-row-button', outline = True, n_clicks = 0,  color = 'light', className ='me-1', style = {'margin-left': 10,'margin-top': 10,'width': 120}),
                                        

                                                        ]),
                                                        dbc.Col(
                                                            html.Div([
                                                                dbc.Button('Download', id = 'download-button', outline = True, n_clicks = 0,  color = 'light', className ='me-1', style = {'margin-left': 10,'margin-top': 10,'width': 120}),
                                                                dcc.Download(id="download-dataframe-csv")
                                                            ]),
                                                        )


                                                                    
                                                    ]),
                                                                    # html.Div(id = 'output-text')

                                ])
                                            
                                        ,
                                        dcc.Tab(
                                            id = 'output-tab', value = 'Derived Rock Properties', style=tab_style, selected_style=tab_selected_style, label = 'Derived Rock Properties', children =  [
                                                    dcc.Loading(id = 'loading-2',   type = 'circle', children = [
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
                            dcc.Tabs(id = 'wavelet-tabs', style ={'margin-top': '10px','margin-bottom': '3px'},  value ='Wavelet Frequency', children  =[
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
                                    dcc.Tab(id = 'phase-shift-tab', label = 'Phase Rotation', 
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
                                dcc.Loading(id = 'loading-wav',   type = 'circle', children = [
                                            
                                            html.Div([dcc.Graph(id = 'wavelet-fig',style=waveletfig_style)]),
                                ])

                        ], style= col_style)
                    ], width =4)

                    ], style = row1_style),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    
                                dcc.Tabs(id = 'tabs-outputs', style ={'margin-top': '10px','margin-bottom': '3px'}, children  =[
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
                                            dcc.Slider(id="scale", tooltip = { 'always_visible': True }, min=.5, max=50, step=.5,value=20, 
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
                                    dcc.Loading(id = 'loading-synthetic',   type = 'circle', children = [
                                                                html.Div([dbc.Label('Synthetic Logs')], style = {'margin-bottom': '0px',
                                                                            'display': 'inline-block',
                                                                            'vertical-align': 'top','color':'white'}),
                                                                dcc.Graph(id = 'output-synthetic',style=logfig_style),
                                                    ])
                                    ,
                                    
                                ])
                            ], style =col_style)
                    ], width =7)

                    , 
                    dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Tabs(id= 'output-fig-tabs', style ={'margin-top': '10px','margin-bottom': '3px'},children = [
                                        dcc.Tab(id ='avo-fig', label ='Amplitude vs Offset',  style=tab_style, selected_style=tab_selected_style,
    children = [
                                            dcc.Loading(id = 'loading-avo',   type = 'circle', children = [
                                                            html.Div([ dcc.Graph(id = 'AvO-plot',style=avofig_style)]),
                                                ])
                                                
                                            ]),
                                            dcc.Tab(id ='eei-fig', label ='Extended Elastic Impedance', style=tab_style, selected_style=tab_selected_style,
    children = [
                    dcc.Loading(id = 'loading-eei',   type = 'circle', children = [
                                                            html.Div([dcc.Graph(id = 'EEI-plot',style=eeifig_style)]),
                                                ])

                                                
                                            ]),
                                            dcc.Tab(id ='ig-fig', label ='Intercept-Gradient', style=tab_style, selected_style=tab_selected_style,
    children = [
                    dcc.Loading(id = 'loading-ig',   type = 'circle', children = [
                                                            html.Div([dcc.Graph(id = 'IG-plot',style=igfig_style)]),
                                                ])

                                                
                                            ]),
                                    ]
                                
                                    ),                              
                            
                                ])
                            , style =col_style)
                        ], width =5),
                    ], style = row2_style),    

                ])
    return app

create_dash_layout(app)


# Run flask app
if __name__ == "__main__": app.run_server(debug=False, host='0.0.0.0', port=1100)