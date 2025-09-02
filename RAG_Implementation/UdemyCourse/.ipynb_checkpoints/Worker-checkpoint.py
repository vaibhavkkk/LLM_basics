# Run command
# conda activate vk-llms
# /opt/anaconda3/envs/vk-llms/bin/python "RAG Implementation/UdemyCourse/Worker.py"

#imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI


# Low Cost LLM model -> gpt-4o-mini

MODEL = "gpt-4o-mini"

# load environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# create openai client
openai = OpenAI(api_key=api_key)

context = {}

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
employees_path = os.path.join(script_dir, "knowledge-base", "employees", "*")

employees = glob.glob(employees_path)

print(f"Looking for employees in: {employees_path}")
print(f"Employees found: {employees}")

for employee in employees:
    # Extract filename without extension
    filename = os.path.basename(employee)
    name = os.path.splitext(filename)[0]  # Remove .md extension

    print(f"Processing employee: {name}")

    doc = ""
    try:
        with open(employee, "r", encoding='utf-8') as f:
            doc = f.read()
            context[name] = doc
            print(f"Successfully loaded: {name}")
    except Exception as e:
        print(f"Error loading {employee}: {e}")

print(f"Total employees loaded: {len(context)}")
print(f"Employee names: {list(context.keys())}")

print(context['Alex Chen'])


## Load Products

products_path = os.path.join(script_dir, "knowledge-base", "products", "*")
products = glob.glob(products_path)

print(f"Looking for products in: {products_path}")
print(f"Products found: {products}")

for product in products:
    # Extract filename without extension
    name = product.split('/')[-1][:-3]
    doc = ""
    try:
        with open(product, "r", encoding='utf-8') as f:
            doc = f.read()
            context[name] = doc
            print(f"Successfully loaded: {name}")
    except Exception as e:
        print(f"Error loading {product}: {e}")

print(f"Total products loaded: {len(context)}")
print(f"Product names: {list(context.keys())}")
print(context['Carllm'])

system_message = "You are expert in answering questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."

def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title in message:
            relevant_context.append(context_details)
    return relevant_context

print("/n Relevant Context : /n ")
print(get_relevant_context("What is Carllm and Who is Oliver Spencer?"))        
    