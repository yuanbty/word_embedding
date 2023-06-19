import json
import glob
import csv
from csv import DictReader, writer


files = glob.glob('/Users/yuantian/Desktop/Financial_Word_Embedding/archive/2018_05_11/*.json', recursive=True)


csv_file = f'/Users/yuantian/Desktop/Financial_Word_Embedding/2018_05_11output.csv'
with open(csv_file, 'w', newline='') as csvfile:
    
        writer = csv.writer(csvfile)
        writer.writerow(['Url', 'Title', 'Text'])
        
        for single_file in files:

            with open(single_file) as f:

                data = json.load(f)

                url = data['url']
                title = data['title']
                text = data['text']

                writer.writerow([url, title, text])