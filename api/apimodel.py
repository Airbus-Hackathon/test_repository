from pydantic import BaseModel
from BART_utilities import *
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import uvicorn
from fastapi import FastAPI, HTTPException
import pytorch_lightning as pl


app = FastAPI()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
model = LitModel.load_from_checkpoint("outputtrainedairbustrainedsavedfull.ckpt", learning_rate=2e-5, tokenizer=tokenizer)

df = pd.read_json('../datasets/trained dataset/airbus_helicopters_train_set.json')
df = df.transpose().reset_index()
df = df[['reference_summary', 'generated_summary']]

class Text(BaseModel):
    text: str
    
@app.post("/summarize/")
async def summarize(request: Text):
    
    try:
        
        inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True)
        generated_text = model.generate_text(inputs, eval_beams=5) 
        
        return {"summary": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)