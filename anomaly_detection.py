import sys, argparse
sys.path.insert(0, 'libs')
import random as rnd
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from AnomalyDetection import AnomalyDetection
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import time


# constants

ANOMALIES_CHECK = True
SENSORS = ['bed', 'sleep', 'social', 'toilet', 'tv', 'pc']
COLORS = ['red', 'pink', 'blue', 'green', 'violet', 'brown']
SIZES = [(6, 8), (5, 8), (5, 8), (7, 6), (3, 9), (4, 8)]
SIZES_2 = [(5, 9), (4, 9), (3, 13), (6, 7), (4, 8), (3, 10)]
DF_NORMAL_FILE = 'data/normal.csv'
DF_NOT_NORMAL_FILE = 'data/not_normal.csv'

# Function that permits to extract a df based on sensor name (only for final analysis)


def subset_df_sensor(df, name):
    return df[df['sensor_name'] == name].reset_index()[['datetime', 'sensor_name', 'sensor_state']]


#Helper functions for normalizing datasets and transforming them

def normalize_df(df):
    df['datetime'] = pd.Series(datetime.datetime.strptime(
        val, '%Y-%m-%d %H:%M:%S') for val in df['datetime'])
    df.loc[df['sensor_state'] == 'ON', ['sensor_state']] = 1
    df.loc[df['sensor_state'] == 'OFF', ['sensor_state']] = 0
    return df

# function that transorms a df into start time and duration
def transform_df(df, sensor, state_to_analyze):
    selected_df = df[df['sensor_name'] == sensor].reset_index()
    starts = []
    durations = []
    anomalies = []
    if selected_df.loc[0, 'sensor_state'] != state_to_analyze:
        selected_df = selected_df.drop(selected_df.index[0])
    if selected_df.loc[len(selected_df)-1, 'sensor_state'] == state_to_analyze:
        selected_df = selected_df.drop(selected_df.index[len(selected_df)-1])
    selected_df = selected_df.reset_index()
    # get starts and durations
    for i in range(0, len(selected_df) - 1, 2):
        starts.append(selected_df.loc[i, 'datetime'])
        duration_diff = selected_df.loc[i+1,
                                        'datetime'] - selected_df.loc[i, 'datetime']
        durations.append((duration_diff.days * 24 * 60 * 60) +
                         duration_diff.seconds)
        if(ANOMALIES_CHECK):
            anomalies.append(selected_df.loc[i, 'anomaly'])

    if(ANOMALIES_CHECK):
        data = {'sensor_name': sensor, 'timestamp': starts,
                'start': starts, 'duration': durations, 'anomaly': anomalies}
    else:
        data = {'sensor_name': sensor, 'timestamp': starts,
                'start': starts, 'duration': durations}
    new_df = pd.DataFrame(data)

    # normalize timestamp
    for index, row in new_df.iterrows():
        time = new_df.loc[index, 'start']
        new_df.loc[index, 'start'] = int(
            time.hour) * 60 * 60 + int(time.minute) * 60 + int(time.second)

    # insert only time info for plots
    new_df['time'] = pd.Series([val.time() for val in new_df['timestamp']])

    return new_df

#Class that creates the SOM and executes the anomaly detection to a given dataset
class SOM_Analysis:
    
    def __init__(self, name, df_normal, df_not_normal, state_to_analyze):
        self.name = name
        self.d = transform_df(df_normal, name, state_to_analyze)
        self.d_n = transform_df(df_not_normal, name, state_to_analyze)
        
    def begin_anomaly_detection(self, size):
        self.scaler = StandardScaler()
        self.train = self.scaler.fit_transform(np.float32(self.d[['start', 'duration']]))
        self.test = self.scaler.transform(np.float32(self.d_n[['start', 'duration']]))

        self.size = size
        self.anomaly_detector = AnomalyDetection(self.size, 2, 0.001, numberOfNeighbors=3, minNumberPerBmu=1)
        
        self.anomaly_detector.fit(self.train, 10000)
        self.normal_indicators = self.anomaly_detector.evaluate(self.train)
        self.threshold = np.percentile(self.normal_indicators, 75)
        #print(self.threshold)
        self.anomaly_metrics = self.anomaly_detector.evaluate(self.test)
        
        self.selector = self.anomaly_metrics >= self.threshold
        self.d_n['predicted'] = 0
        self.d_n.loc[self.selector, 'predicted'] = 1
        
        
        if ANOMALIES_CHECK:
            self.c = pd.crosstab(self.d_n['anomaly'], self.d_n['predicted'])
            
    def get_detector():
        return self.anomaly_detector

#Analysis for non-actions
class Outlier_Analysis:
    
    def __init__(self, name, df_normal, df_not_normal):
        self.name = name
        self.d = transform_df(df_normal, name, 0)
        self.d_n = transform_df(df_not_normal, name, 0)
        
    def begin_anomaly_detection(self):
        self.train = self.d[ 'duration']
        self.test = self.d_n[ 'duration']
        self.q3 = np.percentile(self.train, 75)
        self.q1 = np.percentile(self.train, 25)
        self.threshold = self.q3 + 1.5*(self.q3 - self.q1)
        selector = self.test > self.threshold
        self.d_n['predicted'] = 0
        self.d_n.loc[selector, 'predicted'] = 1

#Adding anomalies to original datasets
def add_anomalies_to_original_df(df, analyzed_df, analyzed_df_2):
    df_temp = pd.DataFrame(columns = ['datetime', 'sensor_name', 'sensor_state'])
    for i in range(0, len(df)):
        date = df.loc[i, 'datetime']
        name = df.loc[i, 'sensor_name']
        state = df.loc[i, 'sensor_state']
        if(state == 0):
            new_state = 1
        else:
            new_state = 0
        line = pd.DataFrame({"datetime": date - datetime.timedelta(seconds=1),
                                 "sensor_name": name,
                                 "sensor_state": new_state}, index = [i])
        df_temp = pd.concat([df_temp, line])

    df = pd.concat([df, df_temp]).sort_values(by=['datetime'])
    df = df[['datetime', 'sensor_name', 'sensor_state']].reset_index()

    for i in range(len(analyzed_df)):
        start = analyzed_df.loc[i, 'timestamp']
        duration = int(analyzed_df.loc[i, 'duration'])
        prediction = analyzed_df.loc[i, 'predicted']
        end = start + datetime.timedelta(seconds = duration)
        condition = ((df['datetime'] >= start) &
                     (df['datetime'] <= end))
        df.loc[condition, 'predicted_1'] = prediction
        
    for i in range(len(analyzed_df_2)):
        start = analyzed_df_2.loc[i, 'timestamp']
        duration = int(analyzed_df_2.loc[i, 'duration'])
        prediction = analyzed_df_2.loc[i, 'predicted']
        end = start + datetime.timedelta(seconds = duration)
        condition = ((df['datetime'] >= start) &
                     (df['datetime'] <= end))
        df.loc[condition, 'predicted_0'] = prediction
    return df

def get_anomaly_boundaries(action_results, non_action_results):
    anomaly_boundaries = []
    for i in range(len(action_results)):
        selected_df = action_results[i].d_n
        for j in range(len(selected_df)):
            if(selected_df.loc[j, 'predicted'] == 1):
                boundaries = {}
                boundaries['sensor_index'] = i + 1
                boundaries['anomaly_color'] = 'LightSeaGreen'
                boundaries['start'] = selected_df.loc[j, 'timestamp']
                boundaries['end'] = selected_df.loc[j, 'timestamp'] + datetime.timedelta(seconds = int(selected_df.loc[j, 'duration']))
                anomaly_boundaries.append(boundaries)
                
    for i in range(len(non_action_results)):
        selected_df = non_action_results[i].d_n
        for j in range(len(selected_df)):
            if(selected_df.loc[j, 'predicted'] == 1):
                boundaries = {}
                boundaries['sensor_index'] = i + 1
                boundaries['anomaly_color'] = 'darkviolet'
                boundaries['start'] = selected_df.loc[j, 'timestamp']
                boundaries['end'] = selected_df.loc[j, 'timestamp'] + datetime.timedelta(seconds = int(selected_df.loc[j, 'duration']))
                anomaly_boundaries.append(boundaries)
    return anomaly_boundaries

#Plot binary data
def plot_datasets(sensor_dfs, action_results, anomaly_boundaries):
    fig = go.Figure()
    fig = make_subplots(
        rows=len(SENSORS), cols=1, shared_xaxes=True, vertical_spacing=0.05
    )

    for i in range(len(SENSORS)):
        fig.add_trace(go.Scatter(
                        x=sensor_dfs[i]['datetime'],
                        y=sensor_dfs[i]['sensor_state'],
                        name=action_results[i].name,
                        line_color= COLORS[i],
                        opacity=0.8),
                        row= i + 1, col=1)


    fig.update_xaxes(range = ['1970-01-01 00:00:00', '1970-01-01 23:59:59'])
    fig.update_yaxes(tick0=0, dtick=1)
    rects = []
    for i in range(len(anomaly_boundaries)):
        shape = go.layout.Shape(
                type="rect",
                xref="x" + str(anomaly_boundaries[i]['sensor_index']),
                yref="y" + str(anomaly_boundaries[i]['sensor_index']),
                x0=anomaly_boundaries[i]['start'],
                y0=0,
                x1=anomaly_boundaries[i]['end'],
                y1=1,
                fillcolor=anomaly_boundaries[i]['anomaly_color'],
                line=dict(
                    color=anomaly_boundaries[i]['anomaly_color'],
                    width=0,
                ),
                layer = 'below',
                opacity=0.5
            )
        rects.append(shape)
        
    fig.update_layout(shapes=rects)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.show()

#Plot for each activity
import plotly.express as px
def plot_activity_bmu(result, description):
    df_anomaly = result.d_n[result.d_n['predicted'] == 1]
    df_not_anomaly = result.d_n[result.d_n['predicted'] == 0]
    df_original = result.d
    #df['metrics'] = result.anomaly_detector.evaluate(result.test)
    trace0 = go.Scatter(
            x = pd.Series(datetime.datetime.fromtimestamp(val) for val in df_anomaly['start']),
            #x = df['start'] / 60 ,
            y = df_anomaly['duration'] / 60,
            marker = dict(
            color = 'rgb(255,0,0)'),
            mode='markers',
            name = 'Dati anomali')
    trace1 = go.Scatter(
            x = pd.Series(datetime.datetime.fromtimestamp(val) for val in df_not_anomaly['start']),
            #x = df['start'] / 60 ,
            y = df_not_anomaly['duration'] / 60,
            marker = dict(
            color = 'rgb(0,0,255)'),
            mode='markers',
            name = 'Dati normali')
    trace2 = go.Scatter(
            x = pd.Series(datetime.datetime.fromtimestamp(val) for val in df_original['start']),
            #x = df['start'] / 60 ,
            y = df_original['duration'] / 60,
            marker = dict(
            color = 'rgb(0,255,0)'),
            mode='markers',
            name = 'Dati di training')
    fig = go.Figure(data = [trace0, trace1, trace2])
    fig.update_xaxes(showgrid=False, zeroline=False, title = "Orario")
    fig.update_yaxes(showgrid=False, zeroline=False, title = "Durata (min)")
    fig.update_layout(title = result.name + ' ' + description)
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='Train a SOM with an initial dataset and get anomalies for another one.')
    parser.add_argument('-td', '--train_dataset', type=str, required=True, help='Dataset for training the algorithm.')
    parser.add_argument('-pd', '--process_dataset', type=str, required=True, help='Dataset to which execute the algorithm and get anomalies.')
    parser.add_argument('-o', '--output_file', type=str, help='Specifies the path where the output will be saved. Default is \'SOM_generated.csv\'.')
    parser.add_argument('-plotd', '--plot_dataset', action='store_true', help='Plots the final dataset with areas covering anomalies.')
    parser.add_argument('-plotbmu', '--plot_bmu',  action='store_true', help='Plots the final dataset for each sensor, comparing its records with SOM\'s BMUs.')
    parser.add_argument('-sa', '--non_action_som_analysis', action='store_true', help='Non action analysis with SOM algorithm. Default is with outlier detection.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Prints the steps taken from the program while running.')
    args = parser.parse_args()

    #program start time
    start = time.time()

    #Generate analysis for all datasets
    if(args.verbose): print("Lettura e normalizzazione dei datasets " + args.train_dataset + " e " + args.process_dataset + ".")
    df_normal = pd.read_csv(args.train_dataset)
    df_normal = normalize_df(df_normal)

    df_not_normal = pd.read_csv(args.process_dataset)
    df_not_normal = normalize_df(df_not_normal)

    if(args.verbose): print("Analisi attivita\' tramite algoritmo SOM.")
    action_results = []
    for i in range(len(SENSORS)):
        SOM_analysis = SOM_Analysis(SENSORS[i], df_normal, df_not_normal, 1)
        SOM_analysis.begin_anomaly_detection(SIZES_2[i])
        action_results.append(SOM_analysis)


    #Analysis for non-activities
    
    non_action_results = []
    if args.non_action_som_analysis and args.verbose: 
        print("Analisi non-attivita\' tramite algoritmo SOM.")
    elif args.verbose:
        print("Analisi non-attivita\' tramite outlier detection.")

    for i in range(len(SENSORS)):
        if args.non_action_som_analysis:
            analysis = SOM_Analysis(SENSORS[i], df_normal, df_not_normal, 0)
            analysis.begin_anomaly_detection(SIZES[i])
        else:
            analysis = Outlier_Analysis(SENSORS[i], df_normal,df_not_normal)
            analysis.begin_anomaly_detection()
        non_action_results.append(analysis)

    #Adding anomalies
    sensor_dfs = []
    for i in range(len(SENSORS)):
        sensor_df = subset_df_sensor(df_not_normal, SENSORS[i])
        sensor_df = add_anomalies_to_original_df(sensor_df, action_results[i].d_n, non_action_results[i].d_n)
        sensor_dfs.append(sensor_df)

    #Adjust dfs to have all plots starting to the same datetime
    if(args.verbose): print("Modifica dei dataset al fine di far coincidere i tempi di inizio e fine.")
    start_date = df_not_normal.loc[0, 'datetime']
    end_date = df_not_normal.loc[len(df_not_normal)-1, 'datetime']

    for i in range(len(sensor_dfs)):
        begin_line = None
        end_line = None
        if sensor_dfs[i].loc[0, 'datetime'] != start_date:
            begin_line = pd.DataFrame({"datetime": start_date,
                                "sensor_name": sensor_dfs[i].loc[0, 'sensor_name'],
                                "sensor_state": sensor_dfs[i].loc[0, 'sensor_state']}, index = [0])
        if sensor_dfs[i].loc[len(sensor_dfs[i])-1, 'datetime'] != end_date:
            end_line = pd.DataFrame({"datetime": end_date,
                                "sensor_name": sensor_dfs[i].loc[len(sensor_dfs[i])-1, 'sensor_name'],
                                "sensor_state": sensor_dfs[i].loc[len(sensor_dfs[i])-1, 'sensor_state']}, index = [len(sensor_dfs[i])+1])
        if not(begin_line is None):
            if not(end_line is None):
                sensor_dfs[i] = pd.concat([begin_line, sensor_dfs[i], end_line])
            else:
                sensor_dfs[i] = pd.concat([begin_line, sensor_dfs[i]])
        else:
            if not(end_line is None):
                sensor_dfs[i] = pd.concat([sensor_dfs[i], end_line])
        sensor_dfs[i] = sensor_dfs[i].reset_index()
        sensor_dfs[i] = sensor_dfs[i][['datetime', 'sensor_name', 'sensor_state', 'predicted_1', 'predicted_0']]


    #Extract time boundaries of each anomaly for the plot
    if(args.verbose): print("Preparazione dataset finale.")
    final = action_results[0].d_n
    for i in range(1, len(action_results)):
        final = pd.concat([final, action_results[i].d_n])
    final = final.sort_values(by = 'timestamp').reset_index()

    if(args.plot_dataset):
        if(args.verbose): print("Apertura di una nuova finestra per la visualizzazione del dataset finale (segnale binario).")
        anomaly_boundaries = get_anomaly_boundaries(action_results, non_action_results)
        plot_datasets(sensor_dfs, action_results, anomaly_boundaries)
    
    if(args.plot_bmu):
        if(args.verbose): print("Apertura di una nuova finestra per la visualizzazione delle singole attivita\' (confronto con i BMU della SOM).")
        for i in range(len(action_results)):
            plot_activity_bmu(action_results[i], ' attivita\'')
        for i in range(len(non_action_results)):
            plot_activity_bmu(non_action_results[i], ' non attivita\'')

    if(args.output_file):
        output = args.output_file
    else:
        output = 'SOM_generated.csv'
    if(args.verbose): print("Esportazione del dataset nel file " + output)

    final['anomaly'] = final['predicted']
    final = final[['sensor_name', 'timestamp', 'time', 'start', 'duration', 'anomaly']]

    final.to_csv(output)
    #Measuring execution time
    duration = time.time() - start
    if(args.verbose): print("Esecuzione conclusa in %s secondi." %duration)

if __name__ == "__main__":
    main()