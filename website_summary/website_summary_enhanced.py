#!/usr/bin/env python3
"""
Website Summary Tool

This script fetches content from a website and generates a summary using AI.
It supports multiple AI providers (OpenAI, Anthropic, Google) and includes
web scraping capabilities.

Requirements:
- Install dependencies from environment.yml
- Set up API keys in .env file

Usage:
    python website_summary.py --url "https://example.com" --provider openai
"""

import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
from urllib.parse import urljoin, urlparse

# Load environment variables
load_dotenv()

class WebsiteSummarizer:
    def __init__(self, provider="openai"):
        """Initialize the summarizer with specified AI provider."""
        self.provider = provider.lower()
        self.setup_ai_client()

    def setup_ai_client(self):
        """Setup AI client based on provider."""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = "gpt-3.5-turbo"
            except ImportError:
                raise ImportError("OpenAI library not installed")

        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.model = "claude-3-haiku-20240307"
            except ImportError:
                raise ImportError("Anthropic library not installed")

        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.client = genai.GenerativeModel('gemini-pro')
                self.model = "gemini-pro"
            except ImportError:
                raise ImportError("Google GenerativeAI library not installed")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def fetch_website_content(self, url):
        """Fetch and parse website content."""
        try:
            print(f"Fetching content from: {url}")

            # Set headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract main content
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

            if main_content:
                text = main_content.get_text()
            else:
                # Fallback to body content
                text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return {
                'title': title_text,
                'content': text,
                'url': url,
                'word_count': len(text.split())
            }

        except requests.RequestException as e:
            raise Exception(f"Error fetching website: {e}")
        except Exception as e:
            raise Exception(f"Error parsing content: {e}")

    def generate_summary(self, content_data, summary_length="medium"):
        """Generate summary using AI provider."""
        title = content_data['title']
        content = content_data['content']
        word_count = content_data['word_count']

        # Truncate content if too long (most APIs have token limits)
        max_chars = 8000  # Approximate token limit consideration
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
            print(f"Content truncated to {max_chars} characters due to length.")

        # Define summary length
        length_map = {
            "short": "in 2-3 sentences",
            "medium": "in 1-2 paragraphs",
            "long": "in 3-4 detailed paragraphs"
        }
        length_instruction = length_map.get(summary_length, "in 1-2 paragraphs")

        prompt = f"""
        Please provide a comprehensive summary of the following website content {length_instruction}.

        Website Title: {title}
        Content Word Count: {word_count}

        Content:
        {content}

        Summary should include:
        - Main topic and purpose
        - Key points and important information
        - Any notable conclusions or takeaways

        Please provide a clear, well-structured summary:
        """

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries of website content."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text

        except Exception as e:
            raise Exception(f"Error generating summary with {self.provider}: {e}")

    def summarize_website(self, url, summary_length="medium"):
        """Main method to summarize a website."""
        try:
            # Fetch content
            content_data = self.fetch_website_content(url)

            # Generate summary
            summary = self.generate_summary(content_data, summary_length)

            return {
                'url': url,
                'title': content_data['title'],
                'word_count': content_data['word_count'],
                'summary': summary,
                'provider': self.provider
            }

        except Exception as e:
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Summarize website content using AI')
    parser.add_argument('--url', required=True, help='Website URL to summarize')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google'],
                       default='openai', help='AI provider to use')
    parser.add_argument('--length', choices=['short', 'medium', 'long'],
                       default='medium', help='Summary length')
    parser.add_argument('--output', help='Output file to save summary')

    args = parser.parse_args()

    # Validate URL
    parsed_url = urlparse(args.url)
    if not parsed_url.scheme:
        args.url = 'https://' + args.url

    try:
        # Initialize summarizer
        summarizer = WebsiteSummarizer(provider=args.provider)

        # Generate summary
        print(f"Summarizing website using {args.provider}...")
        result = summarizer.summarize_website(args.url, args.length)

        if 'error' in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Display results
        print("\n" + "="*60)
        print(f"WEBSITE SUMMARY")
        print("="*60)
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Word Count: {result['word_count']}")
        print(f"AI Provider: {result['provider']}")
        print(f"Summary Length: {args.length}")
        print("\n" + "-"*60)
        print("SUMMARY:")
        print("-"*60)
        print(result['summary'])
        print("\n" + "="*60)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Website Summary\n")
                f.write(f"URL: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Word Count: {result['word_count']}\n")
                f.write(f"AI Provider: {result['provider']}\n\n")
                f.write(f"Summary:\n{result['summary']}\n")
            print(f"Summary saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()