import pandas as pd
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
data = pd.read_csv('training_data.csv')

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Prepare the dataset
def preprocess_data(examples):
    inputs = [ex for ex in examples['input_text']]
    targets = [ex for ex in examples['output_text']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Define training arguments (without wandb)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=2,
    report_to=[],  # Disable wandb logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model locally
model.save_pretrained('sanitizer_model')
tokenizer.save_pretrained('sanitizer_model')

# Save model as pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('explicit_to_non_explicit_model')
tokenizer = T5Tokenizer.from_pretrained('explicit_to_non_explicit_model')

# Function to replace explicit words with non-explicit words
def replace_explicit_words(text):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Generate the output text
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

    # Decode the output ids to get the text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Test the model with new input sentences

input_text = input("Enter your String: ")
test_sentences = re.split(r'(?<=[.!?])\s+', input_text)

output_text = ""  # Initialize output_text before using it

for sentence in test_sentences:
    cleaned_sentence = replace_explicit_words(sentence)
    output_text += " " + cleaned_sentence  # Append cleaned sentence with a space

output_text = output_text.strip()  # Remove any leading space

print(f"Cleaned: {output_text}")