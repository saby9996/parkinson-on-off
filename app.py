# Developed By Code|<Ill at 10/12/2019
# Developed VM IP 203.241.246.158

#Scikit LEarn and Pandas Imports
import pandas as pd

#Dash Dependencies
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import dash_table

#Plotly Imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.tools.set_credentials_file(username='c.sabyasachi99', api_key='y5FSl1jIheriCgKbK3Ff')

#Self Declared Modules and Packages
from Utilities import helper_functions
from Utilities import gait_algorithm
from Utilities import feature_routine_2
from Model import Model


# External CSS
external_stylesheets=["assets/template.css", "assets/bootstrap.min.css"]
# External Scripts
external_scripts=["assets/bootstrap.min.js"]

# Initializing the Default Constructor of Dash Framework and the Application
app=dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)

server = app.server

app.layout=html.Div([
    #Banner
    html.Div(className='header', children=[
        html.Div(className='header-content', children=[
            html.Div(className='container',children=[
                html.Div(className='row toptown',children=[
                    html.Div(className='col-lg-2', children=[
                        html.Img(src="https://i.ibb.co/513BZkn/Bigger-Trans.png", id='inje_logo')
                    ]),
                    html.Div(className='col-lg-8', children=[
                        html.H2("Parkinson's Disease [On - Off]", id='title'),
                    ]),
                    html.Div(className='col-lg-2', children=[
                        html.Img(src="https://i.ibb.co/513BZkn/Bigger-Trans.png", id='ida_logo')
                    ])


                ])
            ])

        ])
    ]),
    #Body
    html.Div(className='body', children=[
        html.Div(className='section-1',children=[
            html.Div(className='container',children=[
                html.Div(className='row', children=[
                    #Col 1 - Sec-1
                    html.Div(className='col-lg-3 white-bg', children=[
                        html.H2("Upload Course Data", id='sub-title', className='selector'),
                        dcc.Upload(
                            className='upload-area',
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files', className='upload-button')
                            ])
                        ),
                        html.H4("BioSignal View Selection", className='selector'),
                        dcc.RadioItems(
                            id='view_choice',
                            options=[
                                {'label':'Right Accelerometer', 'value':'Right_Accl'},
                                {'label':'Left Accelerometer', 'value':'Left_Accl'}
                            ],
                            value='Right_Accl',
                            className='radio_buttons'
                        ),
                        html.H4("Model Initialization", className='selector'),
                        html.Div(className='button-holder',children=[
                            html.Button('Run Model', id='click_button_1',className='button')
                        ])
                    ]),

                    #Col 2 - Sec-1
                    html.Div(className='col-lg-3 white-bg',children=[
                        html.H4("Input Data Table", className='selector'),
                        html.Div(id='Output-Data-Table')
                    ]),

                    #Col 3 - Sec-1
                    html.Div(className='col-lg-6 white-bg graph', children=[
                        html.H4("Visual Representation of Data", className='selector'),
                        dcc.Graph(id='axis_area',style={'width':'98%','float':'right', 'height':'300px','text-align':'center','position':'relative'})
                    ])
                ])
            ])
        ]),
############################################################################################ Section 2 ############################################################################################################
        html.Div(className='section-2',children=[
            html.Div(className='container',children=[
                html.Div(className='row', children=[
                    html.Div(className='col-lg-5 white-bg', children=[
                        html.H4("Gait Parameters", className='selector'),
                        html.Div(id='Gait-Data-Table')
                    ]),
                    html.Div(className='col-lg-3 white-bg', children=[
                        html.H4("Learning Model Results", className='selector'),
                        #dcc.Graph(id='dist_area',style={'width': '98%', 'float': 'right', 'height': '300px', 'text-align': 'center','position': 'relative'})
                        html.Div(id='ml-results')
                    ]),
                    html.Div(className='col-lg-4 white-bg', children=[
                        html.H4("Receiver Operator Characteristic", className='selector'),
                            html.Div(className='helix',children=[
                                html.Img(src="https://i.ibb.co/WGWrMJT/auc.png",className='auc')
                            ])

                    ])
                ])
            ])
        ])
    ])
], style={'background-color':'#f5f5f5'})


################################################################################## Call Back Functions #######################################################################################################
## 1. Call Back For Visual Representation Of Data
@app.callback(
    Output('axis_area','figure'),
    [Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    Input('view_choice','value')])

def update_graph(contents, filename, view):
    if contents is not None:
        frame=helper_functions.parse_contents(contents, filename)
        if frame is not None:

            if(view=='Right_Accl'):
                #Filtering the Signals
                frame['Right_Accl_X'] = frame['Right_Accl_X'] / abs(frame['Right_Accl_X'].max())
                frame['Right_Accl_Y'] = frame['Right_Accl_Y'] / abs(frame['Right_Accl_Y'].max())
                frame['Right_Accl_Z'] = frame['Right_Accl_Z'] / abs(frame['Right_Accl_Z'].max())

                frame['Right_Accl_X'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Right_Accl_X'])
                frame['Right_Accl_Y'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Right_Accl_Y'])
                frame['Right_Accl_Z'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Right_Accl_Z'])

                # Resultant Calculation
                resultant_value = helper_functions.resultant(frame['Right_Accl_X'], frame['Right_Accl_Y'], frame['Right_Accl_Z'])

                return {
                    'data': [go.Scatter(
                        x=list(range(len(list(resultant_value)))),
                        y=list(resultant_value),
                        mode='lines',
                        line=dict(
                            color='#03B5AA',
                            width=2)
                    )],
                    'layout': go.Layout(
                        xaxis={'title': 'Time Epoch'},
                        yaxis={'title': 'Value'},
                        showlegend=False,
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                        hovermode='closest'
                    )
                }
            elif(view=='Left_Accl'):
                frame['Left_Accl_X'] = frame['Left_Accl_X'] / abs(frame['Left_Accl_X'].max())
                frame['Left_Accl_Y'] = frame['Left_Accl_Y'] / abs(frame['Left_Accl_Y'].max())
                frame['Left_Accl_Z'] = frame['Left_Accl_Z'] / abs(frame['Left_Accl_Z'].max())

                frame['Left_Accl_X'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Left_Accl_X'])
                frame['Left_Accl_Y'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Left_Accl_Y'])
                frame['Left_Accl_Z'] = helper_functions.butter_worth_lowpass(4, 0.9375, frame['Left_Accl_Z'])

                # Resultant Calculation
                resultant_value = helper_functions.resultant(frame['Left_Accl_X'], frame['Left_Accl_Y'], frame['Left_Accl_Z'])
                return {
                    'data': [go.Scatter(
                        x=list(range(len(list(resultant_value)))),
                        y=list(resultant_value),
                        mode='lines',
                        line=dict(
                            color='#D1495B',
                            width=2)
                    )],
                    'layout': go.Layout(
                        xaxis={'title': 'Time Epoch'},
                        yaxis={'title': 'Value'},
                        showlegend=False,
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                        hovermode='closest'
                    )
                }
            else:
                return {
                    'data': [],
                    'layout': go.Layout(
                        xaxis={
                            'showticklabels': False,
                            'ticks': '',
                            'showgrid': False,
                            'zeroline': False
                        },

                        yaxis={
                            'showticklabels': False,
                            'ticks': '',
                            'showgrid': False,
                            'zeroline': False
                        }
                    )
                }
        else:
            return {
                'data': [],
                'layout': go.Layout(
                    xaxis={
                        'showticklabels': False,
                        'ticks': '',
                        'showgrid': False,
                        'zeroline': False
                    },

                    yaxis={
                        'showticklabels': False,
                        'ticks': '',
                        'showgrid': False,
                        'zeroline': False
                    }
                )
            }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                xaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                },

                yaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                }
            )
        }



## 2. Call Back For Updating the Data Table
@app.callback(
    Output('Output-Data-Table','children'),
    [Input('upload-data', 'contents'),
    Input('upload-data', 'filename')]
)
def update_table(contents, filename):
    print(contents)
    print(filename)
    if contents is not None:
        frame = helper_functions.parse_contents(contents, filename)
        if frame is not None:
            # Rename the Columns of data
            return dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in frame.columns],
                data=frame.to_dict("rows"),
                style_table={'overflowX': 'scroll', 'overflowY': 'scroll', 'maxHeight': '300px',
                             'border': 'thin #003F5C solid'},
                style_cell={'textAlign': 'center', 'width': '100px'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[{
                    'if': {'column_id': 'Label'},
                    'backgroundColor': '#3D9970',
                    'color': 'white'}
                ]
            )
        else:
            return html.Div()
    else:
        return html.Div()

## 3. Call Back For Fetching Gait Parameters
@app.callback(
    Output('Gait-Data-Table','children'),
    [Input('upload-data', 'contents'),
    Input('upload-data', 'filename')]
)
def update_table(contents, filename):
    if contents is not None:
        frame = helper_functions.parse_contents(contents, filename)
        if frame is not None:
            Stride_Vel_Left, Step_Vel_Left, Stride_Len_Left, Step_Len_Left, Stride_Time_Left, Step_Time_Left = gait_algorithm.gait_params(frame['Left_Accl_X'], frame['Left_Accl_Y'], frame['Left_Accl_Z'])
            Stride_Vel_Right, Step_Vel_Right, Stride_Len_Right, Step_Len_Right, Stride_Time_Right, Step_Time_Right = gait_algorithm.gait_params(frame['Right_Accl_X'], frame['Right_Accl_Y'], frame['Right_Accl_Z'])

            gait={'Gate Parameters':['Stride Velocity','Step Velocity','Stride Length','Step Length','Stride Time','Step Time'],
                  'Left Leg':[Stride_Vel_Left, Step_Vel_Left, Stride_Len_Left, Step_Len_Left, Stride_Time_Left, Step_Time_Left],
                  'Right Leg':[Stride_Vel_Right, Step_Vel_Right, Stride_Len_Right, Step_Len_Right, Stride_Time_Right, Step_Time_Right]}
            gait_frame=pd.DataFrame(data=gait)
            print(gait_frame)


            return dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in gait_frame.columns],
                data=gait_frame.to_dict("rows"),
                style_table={'overflowX': 'scroll', 'overflowY': 'scroll', 'maxHeight': '300px',
                            'border': 'thin #003F5C solid'},
                style_cell={'textAlign': 'center', 'width': '100px','height':'35px'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[{
                    'if': {'column_id': 'Label'},
                    'backgroundColor': '#3D9970',
                    'color': 'white'}
                ]
            )
        else:
            return html.Div()
    else:
        return html.Div()

# 4 Call Back For the Machine LEarning Results Part
@app.callback(
    Output('ml-results','children'),
    [Input('click_button_1','n_clicks')],
    [State('upload-data', 'contents'),
    State('upload-data', 'filename')]
)
def Evaluation(n_clicks, contents, filename):

    #Checking Click
    if n_clicks!=None:
        #Checking the Contents Structure
        if contents is not None:
            frame = helper_functions.parse_contents(contents, filename)

            if frame is not None:
                feature_frame=feature_routine_2.mega_process(frame)
                prediction_mode, off_probability, on_probability = Model.model_call(feature_frame)
                prediction_ripple=helper_functions.ripple(filename,prediction_mode)
                if prediction_mode=='Off':
                    main_probability=off_probability
                else:
                    main_probability=on_probability

                main_probability=round(main_probability*100,2)

                return \
                    html.Div(className='container', children=[
                        html.Div(className='row', children=[
                            html.Div(className='boxy',children=[
                                html.H4(prediction_ripple + " State",className='boxy-main'),
                                html.H4("State of The Patient",className='boxy-low')
                            ])
                        ]),
                        html.Div(className='row', children=[
                            html.Div(className='boxy', children=[
                                html.H4(main_probability, className='boxy-main'),
                                html.H4("Confidence Level",className='boxy-low',style={'padding-right':'6px','padding-left':'6px'})
                            ])
                        ])
                    ])

            else:
                return html.Div()
        else:
            return html.Div()




if __name__ == '__main__':
    app.run_server(debug=True,port=8090)