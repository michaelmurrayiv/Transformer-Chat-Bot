import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import warnings

debug = False

if not debug:
  warnings.filterwarnings("ignore")
  logging.set_verbosity_error()

# load model, test a user input
enc = LabelBinarizer()
enc.fit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
menu_df = pd.read_csv('input.csv')
enc.fit(menu_df['Label'])

model_ckpt = "bert-base-uncased"
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
model.load_state_dict(torch.load('my_model_prompting.pth', map_location=torch.device('cpu')))
tokenizer = AutoTokenizer.from_pretrained('my_tokenizer_prompting')

text = input(">> ")
inputs = tokenizer(text, return_tensors="pt")

# Get model output
outputs = model(**inputs)

# Process output
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)
print(logits,predictions,enc.inverse_transform(logits.cpu().detach().numpy()))

print("")