import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_data():
  tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
  model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
  return tokenizer, model



tokenizer, model = load_data()



st.title("Chatbot in real time")
from PIL import Image
image = Image.open('Bot.jpg')
st.image(image, caption='Bot')


st.write("This is a demo of a chatbot that uses a pretrained model from the huggingface library.")


st.write("Write a text message as if writing a text message to a human. The machine will attempt to respond with an appropriate text message.")


input = st.text_input('Your text message:')

if 'count' not in st.session_state or st.session_state.count == 6:
  st.session_state.count = 0 
  st.session_state.chat_history_ids = None
  st.session_state.old_response = ''
else:
  st.session_state.count += 1

# tokenize the new input sentence
new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

# append the new user input tokens to the chat history
bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

# generate a response 
st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

# convert the tokens to text, and then split the responses into lines
response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if st.session_state.old_response == response:
  bot_input_ids = new_user_input_ids
  st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
  response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
  
st.write(f"Machine text message: {response}")
st.session_state.old_response = response