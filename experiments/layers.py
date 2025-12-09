from bertviz import model_view
from transformers import AutoModel, AutoTokenizer, utils

utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "Skywork/Skywork-o1-Open-Llama3.1-8B"
input_text = "What is 2+2?"
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view
