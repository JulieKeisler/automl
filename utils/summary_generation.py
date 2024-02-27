import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from utils.config import exp_configs
from dragon.utils.tools import logger

def generate_notebook(config):
    nb = nbf.v4.new_notebook()
    exp_config = exp_configs[config['config']]
    text = f"""\
    # EnergyDragon output analysis
    Results analysis for EnergyDragon to the file {config['filename']} using the {config['config']} config. 
    ### Model plot
    """

    code = f"""\
    import pandas as pd

    path = '{config['results_path']+'/model/method_best'}'
    freq = {exp_config['Freq']}

    architecture = pd.read_csv(path + '/best_model_archi.csv', sep=';')
    architecture"""

    code_plot = """\
    from dragon.utils.plot_functions import draw_cell
    import graphviz
    import numpy as np

    G = graphviz.Digraph(path+'/model_plot.pdf', format='pdf',
                                node_attr={'nodesep': '0.02', 'shape': 'box', 'rankstep': '0.02', 'fontsize': '20', 'fontname': 'sans-serif'})
    n_ts = eval(architecture['ST Cell'][0].split(':')[1].split('|')[0])
    m_ts = np.array(eval(architecture['ST Cell'][0].split(':')[-1]))
    n_f = eval(architecture['Feed Cell'][0].split(':')[1].split('|')[0])
    m_f = np.array(eval(architecture['Feed Cell'][0].split(':')[-1]))

    nn_output = int(architecture['NN Output'][0])
    act1 = architecture['NN Activation 1'][0]
    act2 = architecture['NN Activation 2'][0]

    G, g_nodes = draw_cell(G, n_ts, m_ts, '#FAE7D6', [], name_input='Input', color_input='#931C5B')
    G.node('Flatten', style='rounded,filled', color='black', fillcolor='#CE1C4E', fontcolor='#ECECEC')
    G.node(','.join(['MLP', str(nn_output), act1]), style='rounded,filled', color='black', fillcolor='#CE1C4E', fontcolor='#ECECEC')
    G.edge(g_nodes[-1], 'Flatten')

    G.edge('Flatten', ','.join(['MLP', str(nn_output), act1]))
    G, g_nodes = draw_cell(G, n_f, m_f, '#FAE7D6', g_nodes, name_input=['MLP', str(nn_output), act1],
                                color_input='#CE1C4E')
    G.node(','.join(['MLP', str(freq), act2]), style='rounded,filled', color='black', fillcolor='#CE1C4E', fontcolor='#ECECEC')
    G.edge(g_nodes[-1], ','.join(['MLP', str(freq), act2]))
    G
    """

    error_code = f"""\
    prediction = pd.read_csv(path+'/best_model_test_outputs.csv')
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    print('MAPE: ', np.round(mean_absolute_percentage_error(prediction['Actual'], prediction['Pred'])*100,2), '%')
    print('MAE: ', int(mean_absolute_error(prediction['Actual'], prediction['Pred'])))
    print('RMSE: ', int(mean_squared_error(prediction['Actual'], prediction['Pred'], squared=False)))
    """

    freq = exp_config['Freq']
    if freq == 48:
        freq = '30min'
    elif freq == 24:
        freq = 'H'
    if isinstance(freq, str):
        general_plot = f"""\
        import plotly.express as px
        freq_n = '{freq}'
        
        dates = pd.date_range('{exp_config['Borders']['Border1s'][2]}', '{exp_config['Borders']['Border2s'][2]}', freq=freq_n)
        prediction['Date'] = dates
        prediction.set_index('Date', inplace=True)
        fig = px.line(prediction, x = prediction.index, y=['Pred', 'Actual'])
        fig.show()
        """
        montlhy_plot = f"""\
        prediction['Monthly MAE'] = abs(prediction['Actual'] - prediction['Pred'])
        prediction['Month'] = prediction.index.month_name()
        mae_by_month = prediction.groupby('Month', as_index=False, sort=False)['Monthly MAE'].mean()
        fig = px.line(mae_by_month, x='Month', y='Monthly MAE', title='MAE Error by Month', markers=True)
        fig.show()
        """

        weekly_plot = f"""\
        prediction['Daily MAE'] = abs(prediction['Actual'] - prediction['Pred'])
        mae_by_day = prediction.groupby(prediction.index.date, sort=False)['Daily MAE'].mean()
        mae_by_day = mae_by_day.to_frame()
        mae_by_day.index = pd.to_datetime(mae_by_day.index)
        mae_by_day['Day'] = mae_by_day.index.day_name()
        fig = px.box(mae_by_day, x='Day', y='Daily MAE', title='MAE Distribution by Day of the Week')
        fig.show()
        """

        hourly_plot_1 = f"""\
        prediction['Error'] = abs(prediction['Actual'] - prediction['Pred'])
        prediction['Hour'] = prediction.index.hour
        fig = px.box(prediction, x='Hour', y='Error', title='Error Distribution by Hour of the day')
        fig.show()
        """

        hourly_plot_2 = f"""\
        prediction['Hourly Error'] = abs(prediction['Actual'] - prediction['Pred'])
        hourly_error = prediction.groupby(prediction.index.hour, as_index=False)[['Hourly Error']].mean()
        fig = px.line(hourly_error, x=hourly_error.index, y='Hourly Error', title='Error Distribution by Hour of the day', markers=True)
        fig.show()
        """


    else:
        general_plot = f"""\
        import plotly.express as px
        fig = px.line(prediction, x = prediction.index, y=['Pred', 'Actual'])
        fig.show()
        """

    
    text_pred = f"""### Prediction analysis"""

    if isinstance(freq, str):
        nb['cells'] = [nbf.v4.new_markdown_cell(text),
                    nbf.v4.new_code_cell(code),
                    nbf.v4.new_code_cell(code_plot),
                    nbf.v4.new_markdown_cell(text_pred),
                    nbf.v4.new_code_cell(error_code),
                    nbf.v4.new_code_cell(general_plot),
                    nbf.v4.new_code_cell(montlhy_plot),
                    nbf.v4.new_code_cell(weekly_plot),
                    nbf.v4.new_code_cell(hourly_plot_1),
                    nbf.v4.new_code_cell(hourly_plot_2)]
    else:
        nb['cells'] = [nbf.v4.new_markdown_cell(text),
                    nbf.v4.new_code_cell(code),
                    nbf.v4.new_code_cell(code_plot),
                    nbf.v4.new_markdown_cell(text_pred),
                    nbf.v4.new_code_cell(error_code),
                    nbf.v4.new_code_cell(general_plot)]
    fname = config['results_path'] + '/model/method_best/summary.ipynb'
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')


    ep.preprocess(nb)

    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    logger.info(f"The notebook has been saved to {fname}.")