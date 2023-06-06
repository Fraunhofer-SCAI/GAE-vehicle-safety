
import dash
import sys
import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import time
import gae.oems as oems


from django_plotly_dash import DjangoDash

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../note_scripts/'))
# import cypher_notebook_nrg as cyNrg

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DjangoDash('nrg_fts_dash')

app.css.append_css({
    "external_url": external_stylesheets
})

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


def set_plots(c_grp, ords, ns, nPid, sims, pids):
    start = time.time()
    oem = oems.oems('CEVT')
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    h_data = {
        "sim": True,
        "c_grPID": False,
        "IE": ':.2f',
        "ti": ':.3f',
        "dt": ':.2f'}
    fig1 = px.scatter_3d(
        df1, x="dt", y="ti", z="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID",
        hover_data=h_data)
    fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig1.update_layout(showlegend=False, height=700)
    fig1.update_traces(marker_size=4)

    fig2 = px.scatter(
        df1, x="ti", y="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID",
        hover_data=h_data)
    fig2.update_layout(showlegend=False, height=400)
    # fig2.update_layout(clickmode='event+select')

    fig3 = px.scatter(
        df1, x="dt", y="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID",
        hover_data=h_data)
    # fig3.update_layout(clickmode='event+select')
    fig3.update_layout(showlegend=False, height=400)

    fig4 = px.scatter(
        df1, x="ti", y="dt", color=c_grp, custom_data=["pic"],
        hover_name="PID",
        hover_data=h_data)
    fig4.update_layout(showlegend=False, height=400)
    print(time.time() - start)
    return fig1, fig2, fig3, fig4


c_grp = 'c_grPID'
# c_grp='c_grOrd'
c_grp = 'c_rls'
ords = 10
ns = 100
nPid = 10
sims = '.*stv0_.*fp3.*,.*stv03_.*fp3.*'
pids = ''
# pids =  '10021520, 10020420, 18620120, 18620080'
pids = '10021520, 10020420, 18620080, 18620120, 55131400, 55132410, 55132590,55021040, 55021060, 18620070, 18620110'  # stv0

pids += '18620090, 18620070, 10021870, 10021320, 55131440, 55131220, 55021060, 55021040'  # fp3 stv03
fig1, fig2, fig3, fig4 = set_plots(c_grp, ords, ns, nPid, sims, pids)

app.layout = html.Div([
    html.Div(className='row', children=[
        html.Div(
            dcc.Graph(id='embd-3d', figure=fig1),
            className='six columns'),
        html.Div(
            className='three columns',
            children=html.Div([
                dcc.Graph(id='embd-2ds-ti', figure=fig2),
                dcc.Graph(id='embd-2ds-dt', figure=fig3)])
        ),
        html.Div(
            dcc.Graph(id='curv', figure=fig4),
            className='three columns'),
        html.Div(
            className='three columns',
            children=[
                html.Pre(id='click-data'),
                html.Pre(id='click-data2')
            ]
        ),
    ]),
    html.Div(
        className='row',
        children=[
            html.Div(
                className='nine columns',
                children=[
                    html.Div(
                        children=[
                            html.Div(className='four columns', children=[
                                dcc.Input(
                                    placeholder='Enter sims list or regex...',
                                    type='text',
                                    value=sims,
                                    id='sims'),
                                dcc.Input(
                                    placeholder='Enter pids list or regex...',
                                    type='text',
                                    value=pids,
                                    id='pids'),
                                dcc.Input(
                                    placeholder='Max No Ord...',
                                    type='number',
                                    value=ords,
                                    id='ords'),
                                dcc.Input(
                                    placeholder='Max No sim...',
                                    type='number',
                                    value=ns,
                                    id='ns'),
                                dcc.Input(
                                    placeholder='Max No PIDs',
                                    type='number',
                                    value=nPid,
                                    id='nPid')
                            ], style={'columnCount': 2}),
                            html.Div(dcc.Dropdown(
                                options=[
                                    {'label': 'Sim Group', 'value': 'c_grSim'},
                                    {'label': 'PID Group', 'value': 'c_grPID'},
                                    {'label': 'Ord Group', 'value': 'c_grOrd'},
                                    {'label': 'Release Group', 'value': 'c_rls'},
                                    {'label': 'Loadcase Group', 'value': 'c_lc'}],
                                value=c_grp, id='c_grp'),
                                className='two columns'),
                            html.Div(children=[dcc.Dropdown(
                                options=[
                                    {'label': 'Top', 'value': 'top'},
                                    {'label': 'Iso', 'value': 'iso'},
                                    {'label': 'Single', 'value': 'iso0'},
                                    {'label': 'vTop', 'value': 'top.mp4'},
                                    {'label': 'vBtm', 'value': 'btm.mp4'},
                                    {'label': 'vFront', 'value': 'front.mp4'},
                                    {'label': 'vRight', 'value': 'right.mp4'},
                                ],
                                value='iso', id='pic_view'),
                                html.Button('Submit', id='update'),
                                html.Button('Reset', id='reset')],
                                className='two columns'),
                            html.Div(
                                className='three columns',
                                children=[
                                    # , style=styles['pre'])]
                                    html.Pre(id='click-data-txt'),
                                    html.Pre(id='click-data-txt2')]  # , style=styles['pre'])]
                            ),
                        ])]),
            html.Div(
                className='three columns',
                children=[
                    dcc.Slider(id='time', min=0, max=13, value=7),
                    html.Pre(id='sTime')
                ]
            )
        ])

])


@app.callback(
    [
        Output('embd-3d', 'figure'),
        Output('embd-2ds-ti', 'figure'),
        Output('embd-2ds-dt', 'figure'),
        Output('curv', 'figure'),
    ],
    [
        Input('c_grp', 'value'),
        Input('ords', 'value'),
        Input('nPid', 'value'),
        Input('ns', 'value'),
        Input('sims', 'value'),
        Input('pids', 'value'),
        Input('update', component_property='n_clicks'),
    ]
)
def update_embd_3d(c_grp, ords, nPid, ns, sims, pids, n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return set_plots(c_grp, ords, ns, nPid, sims, pids)


@app.callback(
    [
        Output('click-data', 'children'),
        Output('click-data-txt', 'children'),
        Output('click-data-txt2', 'children'),
        Output('sTime', 'children'),
        Output('embd-2ds-ti', 'clickData'),
        Output('embd-2ds-dt', 'clickData'),
        Output('embd-3d', 'clickData'),
        Output('embd-2ds-ti', 'selectedData'),
        Output('embd-2ds-dt', 'selectedData'),
        Output('reset', component_property='n_clicks'),
    ],
    [
        Input('embd-3d', 'clickData'),
        Input('embd-2ds-ti', 'clickData'),
        Input('embd-2ds-dt', 'clickData'),
        Input('embd-2ds-ti', 'selectedData'),
        Input('embd-2ds-dt', 'selectedData'),
        Input('pic_view', 'value'),
        Input('click-data-txt2', 'children'),
        Input('time', 'value'),
        Input('sTime', 'children'),
        Input('reset', component_property='n_clicks'),
    ])
def display_click_data(cData1, cData2, cData3, sData2, sData3, pos, old, tVal, sTime, reset):
    c_i = None
    if cData2:
        c_i = cData2['points'][0]['customdata'][0]
    if cData3:
        c_i = cData3['points'][0]['customdata'][0]

    pids, sims = [], []
    if sData2:
        sims = [p['customdata'][1] for p in sData2['points']]
        pids = ([p['hovertext'] for p in sData2['points']])
    if sData3:
        sims = [p['customdata'][1] for p in sData3['points']]
        pids = ([p['hovertext'] for p in sData3['points']])

    txt = 'sims: {0}\npids: {1}'.format(
        ', '.join(sims),
        ', '.join(pids)
    )

    if old:
        sims, pids = old.split('\n')
    else:
        sims, pids = ' ', ' '
    if cData1:
        pid_i = str(cData1['points'][0]['hovertext'])
        sim_i = cData1['points'][0]['customdata'][1]
        c_i = cData1['points'][0]['customdata'][0]
        if not sim_i in sims:
            sims += (sim_i + ', ')

        if not pid_i in pids:
            pids += (pid_i + ', ')

    txt2 = '{0}\n{1}'.format(sims, pids)
    if not reset is None:
        txt2 = ''
        txt = 'sims: \npids:'

    if sTime and not c_i:
        try:
            sim_i = sTime.split('/')[2]
            sim_i = '_'.join(sim_i.split('_')[:-2])
        except IndexError:
            sTime_1 = sTime.split('/')[1]
            sim_i = '_'.join(sTime_1.split('_')[:-2])
            pass
        c_i = '_'.join(sTime.split('_')[:-1] + ['iso.png'])
    if c_i:
        if 'mp4' in pos:
            extn = '_{}.png'.format(tVal)
            name = 'CEVT/Vidp/' + sim_i + '_' + pos.replace('.mp4', extn)
            obj = html.Img(
                src=app.get_asset_url(name),
                style={"height": "50vh", "display": "block", "margin": "auto"}
                # , controls=True, autoPlay=True, #, type='video/mp4'
            ),
        else:
            name = c_i.replace('iso', pos)
            obj = html.Img(
                src=name,
                # src=app.get_asset_url(name),
                style={"height": "40vh", "display": "block", "margin": "auto"}
            ),
        return(obj, txt, txt2, name, None, None, None, None, None, None)
    return(None, txt, txt2, None, None, None, None, None, None, None)


if __name__ == '__main__':
    app.run_server(8052, debug=False)
