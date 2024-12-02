#importing necessary libraries
from transformers import BertForSequenceClassification, Trainer,TrainingArguments
import torch 
from preprocess import load_and_preprocess_data, tokenize_data

#loading the pre trained model 

def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',  num_labels=2)
    return model

def train_model():
    data = load_and_preprocess_data()
    tokenized_data = tokenize_data(data)
    num_labels = 2 # number of labels in the dataset
    model = load_model(num_labels)
    train_data = torch.utils.data.TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(data['Label']))


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()
    model.save_pretrained('./results')
    return model