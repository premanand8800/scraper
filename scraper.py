import streamlit as st
import logging
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
import markdown
import re
import os
from groq import Groq
from datetime import datetime
import base64
import hashlib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Groq client (replace with your actual API key)
groq_client = Groq(api_key="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")

class AdvancedUniqueContentScraper:
    def __init__(self, start_url_or_query, max_pages=100, max_depth=3, similarity_threshold=0.8):
        self.start_url_or_query = start_url_or_query
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.visited = set()
        self.to_visit = set()
        self.organized_content = []
        self.output_dir = self.create_output_directory()
        self.content_hashes = set()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.content_vectors = []

    def create_output_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"scraper_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def hash_content(self, content):
        return hashlib.md5(content.encode()).hexdigest()

    def is_content_unique(self, content):
        content_hash = self.hash_content(content)
        if content_hash in self.content_hashes:
            return False
        
        if not self.content_vectors:
            self.content_hashes.add(content_hash)
            return True
        
        content_vector = self.vectorizer.transform([content])
        similarities = cosine_similarity(content_vector, self.content_vectors)
        if np.max(similarities) < self.similarity_threshold:
            self.content_hashes.add(content_hash)
            self.content_vectors = self.vectorizer.fit_transform(self.organized_content + [content])
            return True
        return False

    def interpret_query(self):
        prompt = f"""
        Given the following user query, determine the most likely website URL to scrape:
        
        Query: {self.start_url_or_query}

        If the query is already a valid URL, return it as is. If it's a natural language query,
        interpret it and provide the most likely website URL to scrape.

        Return your answer in the following format:
        URL: [The determined URL]
        Explanation: [A brief explanation of your interpretation]
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that interprets user queries and determines appropriate website URLs for web scraping."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )

        result = response.choices[0].message.content
        url_match = re.search(r"URL: (https?://\S+)", result)
        if url_match:
            return url_match.group(1)
        else:
            raise ValueError("Could not determine a valid URL from the query.")

    def scrape_page(self, url, depth=0):
        if depth > self.max_depth:
            return

        url, _ = urldefrag(url)

        if url in self.visited:
            return

        self.visited.add(url)
        st.text(f"Scraping: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else url
            full_content = self.extract_full_content(soup)
            
            if self.is_content_unique(full_content):
                code_blocks = self.extract_code_blocks(soup)
                organized_content = self.process_with_groq(title, full_content, code_blocks, url)
                self.organized_content.append(organized_content)
            else:
                st.info(f"Similar content already exists. Skipping {url}")
            
            if depth < self.max_depth:
                new_links = self.extract_links(soup, url)
                for link in new_links:
                    if link not in self.visited and len(self.visited) < self.max_pages:
                        self.scrape_page(link, depth + 1)
            
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

    def extract_full_content(self, soup):
        for unwanted in soup.find_all(['script', 'style', 'nav', 'footer']):
            unwanted.decompose()
        return soup.get_text(separator='\n', strip=True)

    def extract_code_blocks(self, soup):
        code_blocks = []
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                language = code.get('class', [''])[0].split('-')[-1] if code.get('class') else ''
                code_content = code.get_text(strip=True)
                code_blocks.append(f"```{language}\n{code_content}\n```\n\n")
        return code_blocks

    def extract_links(self, soup, base_url):
        links = set()
        for a in soup.find_all('a', href=True):
            href = urljoin(base_url, a['href'])
            parsed_href = urlparse(href)
            if parsed_href.netloc == urlparse(base_url).netloc:
                links.add(href)
        return links

    def process_with_groq(self, title, full_content, code_blocks, url):
        prompt = f"""
        Review, edit, and organize the following full content from the webpage "{title}" (URL: {url}):

        Main Content:
        {full_content[:30000]}

        Code Blocks:
        {"".join(code_blocks)[:10000]}

        Please organize this content without summarization. Your task is to:
        1. Correct any grammatical or spelling errors.
        2. Improve readability by breaking long paragraphs into shorter ones where appropriate.
        3. Add appropriate headers (##, ###, etc.) to structure the content.
        4. Use bullet points or numbered lists for any series of items or steps.
        5. Ensure code blocks are properly formatted and labeled.
        6. Add a brief introduction at the beginning and a conclusion at the end.
        7. Maintain all original information - do not remove any content.
        8. Ensure the content is unique and not repetitive with previously processed content.

        Format the output in Markdown, using appropriate headers, lists, and code blocks.
        Begin the content with the title of the page as a level 1 header (# Title).
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a proficient editor and content organizer. Your task is to improve and structure web content without summarizing or removing information."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )

        return response.choices[0].message.content

    def scrape_site(self):
        start_url = self.interpret_query()
        self.scrape_page(start_url)
        return '\n\n---\n\n'.join(self.organized_content)

    def save_markdown(self, content, filename='output.md'):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        st.success(f"Markdown content saved to {file_path}")
        return file_path

    def convert_to_html(self, markdown_file, html_file='output.html'):
        html_path = os.path.join(self.output_dir, html_file)
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = markdown.markdown(markdown_content, extensions=['tables'])
        
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        st.success(f"HTML file created: {html_path}")
        return html_path

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("Advanced Unique Content Scraper")

    user_input = st.text_input("Enter a URL or a query about what you want to scrape:")
    max_pages = st.number_input("Enter the maximum number of pages to scrape:", min_value=1, value=10)
    max_depth = st.number_input("Enter the maximum depth to scrape:", min_value=1, value=3)
    similarity_threshold = st.slider("Set similarity threshold (higher value means more unique content):", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    if st.button("Start Scraping"):
        scraper = AdvancedUniqueContentScraper(user_input, max_pages=max_pages, max_depth=max_depth, similarity_threshold=similarity_threshold)

        with st.spinner("Scraping in progress..."):
            content = scraper.scrape_site()

        if content:
            markdown_path = scraper.save_markdown(content)
            html_path = scraper.convert_to_html(markdown_path)

            st.success("Scraping completed!")
            st.markdown(get_binary_file_downloader_html(markdown_path, 'Markdown'), unsafe_allow_html=True)
            st.markdown(get_binary_file_downloader_html(html_path, 'HTML'), unsafe_allow_html=True)

            with st.expander("Preview Content"):
                st.markdown(content)
        else:
            st.error("No unique content was scraped. The process might have encountered an error or all content was too similar.")

if __name__ == "__main__":
    main()