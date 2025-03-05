import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('results/final_model')
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

category_mapping = {
    "体育": 0,
    "财经": 1,
    "房产": 2,
    "家居": 3,
    "教育": 4,
    "科技": 5,
    "时尚": 6,
    "时政": 7,
    "游戏": 8,
    "娱乐": 9,
}

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits 
    
    predictions = torch.softmax(logits, dim=-1)
    
    result = {}
    for i in range(len(predictions[0])): 
        predicted_class_label = [key for key, value in category_mapping.items() if value ==i][0]
        result[predicted_class_label] = float(predictions[0][i])
    
    return result

iface = gr.Interface(fn=predict, inputs="text", outputs="label")

iface.launch(share=True)