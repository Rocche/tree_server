import requests
import json
import sys, argparse, time
import pandas as pd
import dateutil.parser
from datetime import date, timedelta

BASE_URL = 'http://tree.essereweb.it/api/rilevamenti/paziente'

def get_dates(start, end):
    dates = []
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    time_diff = end_date - start_date
    #have to include end date, so add 1 day to timedelta
    for i in range(time_diff.days + 1):
        d = start_date + timedelta(days=i)
        dates.append(date.strftime(d, "%Y-%m-%d"))
    return dates

def main():
    parser = argparse.ArgumentParser(description='Ritorna il dataset dei rilevamenti di un paziente dato id, data di inizio e data di fine.')
    parser.add_argument('-i', '--id', type=str, required=True, help='ID del paziente.')
    parser.add_argument('-s', '--start', type=str, required=True, help='Data di inizio (formato YYYY-mm-dd, ad esempio 2020-10-28).')
    parser.add_argument('-e', '--end', type=str, required=True, help='Data di fine (formato YYYY-mm-dd, ad esempio 2020-10-28).')
    parser.add_argument('-o', '--output_file', type=str, help='Specifica il percorso dove salvare il file. Default \'rilevamento_[ID]_[inizio]_[fine].csv\'.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Stampa i passi eseguiti dal programma.')
    args = parser.parse_args()
    
    #program start time
    program_start = time.time()
    final = []
    
    start = args.start
    end = args.end
    id = args.id
    
    #generates dates
    if(args.verbose): print("Estrapolazione date necessarie per le chiamate al server.")
    dates = get_dates(start, end)
    
    for d in dates:
        if(args.verbose): print("Download rilevamenti del " + d +".")
        req = requests.get(BASE_URL + '?paziente_id=' + id + '&datetime=' + d)
        j = json.loads(req.text)
        if(args.verbose and len(j) > 0): print("Dati disponibili. Caricamento nel dataset.")
        if(args.verbose and len(j) == 0): print("Dati non disponibili.")
        for rilevamento in j:
            values = rilevamento['values']
            for sensor_event in values:
                event = {}
                event['sensor_name'] = sensor_event['attribute']['name']
                event['timestamp'] = dateutil.parser.parse(sensor_event['timestamp'])
                event['sensor_state'] = sensor_event['value']
                final.append(event)
                
    df = pd.DataFrame(final).sort_values(by='timestamp').reset_index()
    
    if(args.output_file):
        output = args.output_file
    else:
        output = 'rilevamento_' + id + "_" + start + "_" + end + ".csv"
        
    if(args.verbose): print("Esportazione del dataset nel file " + output)
    df[['timestamp', 'sensor_name', 'sensor_state']].to_csv(output)
    
    #Measuring execution time
    duration = time.time() - program_start
    if(args.verbose): print("Esecuzione conclusa in %s secondi." %duration)



if __name__ == "__main__":
    main()