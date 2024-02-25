from pydantic import BaseModel
from BART_utilities import *
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import uvicorn
from fastapi import FastAPI, HTTPException
import pytorch_lightning as pl
import subprocess

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
loaded_model = LitModel.load_from_checkpoint("outputtrainedairbustrainedsavedfull.ckpt", learning_rate=2e-5, tokenizer=tokenizer)

airbus = pd.read_json('../../../dataset/legal_summarization/airbus_helicopters_train_set.json')
airbus = airbus.transpose().reset_index()
airbus_train = airbus[['original_text', 'reference_summary']]

airbus_test = pd.read_json('../datasets/test/test_set.json')
airbus_test = airbus_test.transpose().reset_index()

generated_summaries = []

for i, row in airbus_test.iterrows():
    inputs = tokenizer(row['original_text'], return_tensors="pt", max_length=512, truncation=True)
    generated_text = loaded_model.generate_text(inputs, eval_beams=5)
    generated_summary = generated_text[0]
    generated_summaries.append(generated_summary)

airbus_test['generated_summary'] = generated_summaries

### generate dataset to evaluate test set
airbus_test.to_json('../datasets/results/jsongenerated.json', orient='records')

commande = ["python", "evaluate.py", "-r", "../datasets/test/test_set.json", "-g", "../datasets/results/jsongenerated.json"]

# Exécuter la commande
subprocess.run(commande)