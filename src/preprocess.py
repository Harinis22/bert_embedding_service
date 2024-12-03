import pandas as pd
from transformers import BertTokenizer

#adjust the path to data folder
def load_and_preprocess_data():
    # Load the data
      # Load the data
    url = "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log_structured.csv"
    df = pd.read_csv(url) 
    data = df[['Content', 'EventId', 'EventTemplate']] # selecting last 3 columns needed
    
    return data

def tokenize_data(data):
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Combine the columns 'Content', 'EventId', and 'EventTemplate' into a single string
    combined_data = data['Content'] + " " + data['EventId'] + " " + data['EventTemplate']
    
    # Tokenize the combined text
    tokenized_data = tokenizer(list(combined_data), padding=True, truncation=True, return_tensors="pt")
    
    return tokenized_data
print(tokenize_data)
