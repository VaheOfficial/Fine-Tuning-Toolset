import os
import json
from bs4 import BeautifulSoup

# Create a folder to store the beautified chapters
if not os.path.exists('chapters'):
    os.makedirs('chapters')

# Load the saved HTML content from the JSON file
with open('detroit_become_human_chapters_content.json', 'r') as f:
    chapters_content = json.load(f)

# Function to beautify HTML content
def beautify_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup.prettify()

# Process each chapter's content
for chapter in chapters_content:
    chapter_title = chapter['chapter']
    beautified_html = beautify_html(chapter['content'])
    
    # Create a filename based on the chapter title
    filename = os.path.join('chapters', f"{chapter_title.replace(' ', '_')}.txt")
    
    # Save the beautified HTML to a text file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(beautified_html)

print("HTML content has been successfully beautified and saved to files.")
