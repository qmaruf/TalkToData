# Talk to Your Data

## Overview

This code provides a simple chat interface that uses OpenAI's GPT-3.5-Turbo to answer questions related to a document.

## Features

- **Upload Documents**: Users can upload PDF or text files.
- **Conversational Interface**: Engage in a chat-like conversation with the AI.
- **Real-time Messages**: Messages from the user and AI are displayed in real-time.

## Usage

1. Set your OpenAI API key in a `.env` file.
```bash
OPENAI_API_KEY=sk-...
```
2. Install the dependencies.
```bash
pip install -r requirements.txt
```
3. Run the app using
```bash
streamlit run chat.py
```
4. Upload a document through the app's sidebar.
5. Start a conversation with the AI by sending a message.

## Deploy to Hugging Face Spaces
huggingface-cli login
git remote add huggingface https://huggingface.co/spaces/qmaruf/talk-to-data


## Author
Quazi Marufur Rahman | maruf.csdu@gmail.com
