import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained('tokenizer-llama_career_0.5.0')
# model = AutoModelForCausalLM.from_pretrained('llama-2-7b-career-0.5.0')

# Function to generate response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("Hugging Face Model Chatbot")
st.write("Interact with your custom model")

input_text = st.text_input("You:", "")

if st.button("Generate Response"):
    if input_text:
        response = generate_response(input_text)
        st.write(f"Model: {response}")
    else:
        st.write("Please enter some text.")
