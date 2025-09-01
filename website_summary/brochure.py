import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# Load environment variables

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key
if not api_key:
    print("❌ No API key was found. Please set OPENAI_API_KEY in your .env file.")
elif not api_key.startswith('sk-'):
    print("❌ The API key format is not valid. OpenAI keys should start with 'sk-'")
elif api_key.strip() != api_key:
    print("❌ The API key contains leading or trailing spaces. Please remove them.")
else: