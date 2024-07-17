import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Loading api key from environment
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_KEY)

memory = ConversationBufferMemory()

def predictor(input, temperature=0.7):
    global memory
    # Storing user input
    memory.chat_memory.messages.append({"role": "user", "content": input})  # Add user message to chat memory
    
    # Converting chat memory to messages format needed for OpenAI API
    messages = memory.chat_memory.messages
    
    # Getting the response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature
    )
    
    reply_content = response.choices[0].message.content.strip()

    # Storing the assistant response
    memory.chat_memory.messages.append({"role": "assistant", "content": reply_content})  # Add assistant message to chat memory
    
    # Formatting response for the Gradio interface
    chat_history = [(memory.chat_memory.messages[i]["content"], memory.chat_memory.messages[i+1]["content"]) for i in range(0, len(memory.chat_memory.messages)-1, 2)]

    return chat_history

def clear_history():
    global memory
    memory = ConversationBufferMemory()
    return []



#Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# My Conversational Chatbot")
    gr.Markdown("""
    ## How to use and interact with the bot
    - Type your message into the textbox and press 'Submit' to chat with the bot.
    - Adjust the temperature slider to control the randomness of the output.
    - Press 'Clear Chat' to reset the chat history.
    
    This bot integrates the  **GPT-4 model** and uses Langchain memory + a Gradio interface.
    """)
    
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here")
        temp_slider = gr.Slider(0, 1, value=0.7, step=0.01, label="Temperature", info="Controls randomness of output")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear Chat")
        
        submit_btn.click(predictor, [txt, temp_slider], chatbot)
        submit_btn.click(lambda: "", None, txt)  # Clear textbox on submit
        clear_btn.click(clear_history, [], chatbot)  # Clear chat history on clear button click

    # Launch full-screen demo
    demo.launch(share=True, width="100%", height="210vh")