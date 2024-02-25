from pydantic import BaseModel
from BART_utilities import *
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import uvicorn
from fastapi import FastAPI, HTTPException
import pytorch_lightning as pl
import subprocess
import json

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

airbus_test = pd.read_json('../datasets/test/test_set.json')
airbus_test = airbus_test.transpose().reset_index()

new_tokens = ['<F>', '<RLC>', '<A>', '<S>', '<P>', '<R>', '<RPC>']
special_tokens_dict = {'additional_special_tokens': new_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
bart_model.resize_token_embeddings(len(tokenizer))

loaded_model = LitModel.load_from_checkpoint("../checkpoints/outputtrainedairbustestsaved.ckpt", learning_rate=2e-5, tokenizer=tokenizer, model=bart_model)


for i, row in airbus_test.iterrows():
    inputs = tokenizer(row['original_text'], return_tensors="pt", max_length=512, truncation=True)
    generated_text = loaded_model.generate_text(inputs, eval_beams=5)
    airbus_test.at[i, 'generated_summary'] = generated_text[0]


airbus_test.set_index('index', inplace=True)
airbus_test = airbus_test.transpose()
data_dict = airbus_test.to_dict()

json_data_generated = {}
for key, value in data_dict.items():
    json_data_generated[key] = {
        'uid': value['uid'],
        'generated_summary': value['generated_summary']
    }

# Écrire le dictionnaire dans un fichier JSON
with open('../datasets/results/jsongenerated.json', 'w') as json_file:
    json.dump(json_data_generated, json_file)
    
commande = ["python", "evaluate.py", "-r", "../datasets/test/test_set.json", "-g", "../datasets/results/jsongenerated.json"]

# Exécuter la commande
subprocess.run(commande)