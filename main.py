import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import warnings
import google.generativeai as genai
import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

debug = False

if not debug:
  warnings.filterwarnings("ignore")
  logging.set_verbosity_error()

def main():
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

  text = input("Enter your request (reading, writing, or exiting)\n>> ")
  inputs = tokenizer(text, return_tensors="pt")

  # Get model output
  outputs = model(**inputs)

  # Process output
  logits = outputs.logits
  #predictions = torch.argmax(logits, dim=1)
  #print(logits,predictions,enc.inverse_transform(logits.cpu().detach().numpy()))

  menu_item = enc.inverse_transform(logits.cpu().detach().numpy())[0].lower()

  if menu_item == 'read':
    read_cover_letter()
  elif menu_item == 'write':
    input_text = input("Enter 1 for sentence autocomplete, 2 for cover letter generation\n>> ")
    if input_text == '1':
      sentence_autocomplete()
    elif input_text == '2':
      generate_cover_letter()
  elif menu_item == 'exit':
    return
  
  main()

def read_cover_letter():
  load_dotenv()

  my_api_key = os.getenv("GOOGLE_API_KEY")
  genai.configure(api_key=my_api_key)
  model = genai.GenerativeModel("gemini-1.5-flash")


  print("Paste your input and press Ctrl+D (or Ctrl+Z on Windows):")
  cover = sys.stdin.read()
 # cover = input("Enter the cover letter")
  # Combine user input for prompt
  prompt = f""" I'm going to give you a cover letter, and I want you to 
  print out 3 things for me. The author's name, the relevant skills the author has,
  and their objective. I want you to print in the format:
  "Name: [author's name]"
  "Skills: [relevant skills]"
  "Objective: [objective]"

  Here is the cover letter: {cover}
  """

  response = model.generate_content(prompt)

  print("\n", "_" * 10, "OUTPUT", "_" * 10)
  print(response.text)
  print("_" * 28)

def sentence_autocomplete():
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
  start = input("Sentence Autocomplete On. Type 'Q' + Enter to Exit. Begin Typing: \n")
  text = ""

  while start != 'Q':
    input_ids = tokenizer.encode(start, return_tensors='pt')

    sample_output = model.generate(
      input_ids,
      do_sample=True,
      max_length=50,
      top_k=50,
      top_p=0.95,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id
    )

    # Decode the output, stopping at the first EOS token
    decoded_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    first_sentence = decoded_text.split('.')[0] + '.'
    text = text.strip() + ' ' + first_sentence
    start = input("\n" + text + " ")

  print("\nYour text is: \n\n", text, "\n\nExited. ")

  return

def generate_cover_letter():
  load_dotenv()

  my_api_key = os.getenv("GOOGLE_API_KEY")
  genai.configure(api_key=my_api_key)
  model = genai.GenerativeModel("gemini-1.5-flash")

  # User Input: Only Job Title
  desc = input("Enter the job title: ")
  skills = input("Enter your relevant skills: ")
  # Combine user input for prompt
  prompt = f""" Write me a cover letter with three body paragraphs for a role with the following job 
  title, and skills. The cover letter should have a first paragraph about the position title, where it was found, 
  and a thesis sentence. The second paragraph should be about my skills that are relevant to the position, which I gave to you. 
  The last paragraph should reiterate interest, thank for time, and invite the employer to contact me. 

  Here is the title: {desc}
  Here are my skills: {skills}
  """

  response = model.generate_content(prompt)

  print("\n", "_" * 10, "OUTPUT", "_" * 10)
  print(response.text)
  print("_" * 28)
  return

if __name__ == "__main__":
  main()

