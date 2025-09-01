import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
try:
    from IPython.display import Markdown, display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(content):
        print(content)
    def Markdown(text):
        return text

from openai import OpenAI

# load environment variable in a file called .env
print("Loading environment variables...")
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# check the key
print("Checking API key...")

if not api_key:
    print("❌ No API key was found. Please set OPENAI_API_KEY in your .env file.")
elif not api_key.startswith('sk-'):
    print("❌ The API key format is not valid. OpenAI keys should start with 'sk-'")
elif api_key.strip() != api_key:
    print("❌ The API key contains leading or trailing spaces. Please remove them.")
else:
    print("✅ The API key appears to be valid!")
    print(f"   Key preview: {api_key[:10]}...{api_key[-4:]}")

    # ❌ Open AI key issue => insufficient_quota => Using Olama as alternative
    # # Test the API key by making a simple request
    # try:
    #     client = OpenAI(api_key=api_key)
    #     # Make a minimal test request
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": "Hello"}],
    #         max_tokens=5
    #     )
    #     print("✅ API key is working! Successfully connected to OpenAI.")
    # except Exception as e:
    #     print(f"❌ API key validation failed: {str(e)}")
    #     print("   Please check if your API key is correct and has sufficient credits.")

    openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    message = "Hellow Llama! This is my first ever message to you! Hi!"
    response = openai.chat.completions.create(
        model="llama3.2",   
        messages=[{"role": "user", "content": message}],
        max_tokens=500,
        temperature=0.3
    )
#    print(response.choices)
    print(response.choices[0].message.content)

# ----------------------------- #


# A class to represent a Webpage

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class Website:
    def __init__(self, url):
        """
        Create this Website object from given url using BeautifulSoup library
        """

        self.url = url
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "nav", "footer", "header"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

ed = Website("https://edwarddonner.com")
print(ed.title)
print(ed.text)

# ----------------------------- #
# System Prompt - tells LLM what tasks are they performing and in what tone should they use

system_prompt = "You are an assistant that analyzes the contents of a website and provides a \
    short summary, ignoring text that might be navigation related. Respond in markdown format."

# ----------------------------- #
# User Prompt - tells LLM what to do
def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\n The contents of this website is as follows; \
        please provide a short summary of this website in markdown. \
            If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

print(user_prompt_for(ed))
print(system_prompt)

# ----------------------------- #
# Message 

#---- TESTING ---
messages = [
    {"role": "system", "content": "You are a snarky assistant"},
    {"role": "user", "content": "What is 2 + 2?"}
]

response = openai.chat.completions.create(model="llama3.2", messages=messages)
print(response.choices[0].message.content)
#---- TESTING ---


def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]
messages_for(ed)


# ----------------------------- #
# Summarization

def summarize(url):
    website = Website(url)
    messages = messages_for(website)
    response = openai.chat.completions.create(model="llama3.2", messages=messages)
    return response.choices[0].message.content

print("\n ** SUMMARY **")
# print(summarize("https://edwarddonner.com"))

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

# display_summary("https://edwarddonner.com")

display_summary("https://cnn.com")