# Conversational-Chatbot-using-Transformers-and-Streamlit

web app:https://napo178-conversational-chatbot-using-transformers-a-main-cs45xs.streamlitapp.com/

To run the app locally, use the following command :-  
streamlit run main.py`  


The webpage should open in the browser automatically.  
If it doesn't, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://localhost:8501/`  


Transformers models:

https://huggingface.co/transformers/v4.4.2/_modules/transformers/models/auto/modeling_auto.html

GPT-2 + SEC2VEC CALL

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer("Hello world")['input_ids']
tokenizer(" Hello world")['input_ids']

