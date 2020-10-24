from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import argparse
from pathlib import Path
import datetime
import os
import random
import string
import nltk


output_dir = "D:\Github_projects\Finbert_20201019\data\sentiment_transcript_all"
model_path = "D:\Github_projects\Finbert_20201019\models\sentiment\phraseBank"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)

existing_files = os.listdir("D:\Github_projects\Finbert_20201019\data\sentiment_transcript_all")



files = os.listdir("D:\Github_projects\Finbert_20201019\data\pure_transcript_all")
files.sort()
for file_name in files:
    try:
        csv_filename = file_name.split(".")[0]+".csv" 
        print(csv_filename)
        if (csv_filename in existing_files): continue
        path = "D:\Github_projects\Finbert_20201019\data\pure_transcript_all\%s" % str(file_name)
        with open(path,'r') as f:
            text = f.read()
            f.close()
        output = str(file_name).split(".")[0] + '.csv'
        predict(text,model,write_to_csv=True,path=os.path.join(output_dir,output))
    except UnicodeDecodeError:
        with open(path,'r', encoding='utf8') as f:
            text = f.read()
            f.close()
        output = str(file_name).split(".")[0] + '.csv'
        predict(text,model,write_to_csv=True,path=os.path.join(output_dir,output))
    #     print("error")


#now = datetime.datetime.now().strftime("predictions_%B-%d-%Y-%I:%M.csv")
#random_filename = ''.join(random.choice(string.ascii_letters) for i in range(10))

