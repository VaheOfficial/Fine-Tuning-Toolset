from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json
import time

# Setup Selenium with Chrome WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless Chrome
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
chrome_options.binary_location = "/usr/bin/google-chrome"  # Replace with the actual path to your Chrome binary if needed

# Specify the path to your ChromeDriver
service = Service('/usr/local/bin/chromedriver/chromedriver')  # Replace with your WebDriver path
driver = webdriver.Chrome(service=service, options=chrome_options)

# Base URL for chapters
base_url = "https://detroitbecometext.github.io/chapters/"

# Initialize an empty list to store dialogues
chapters_content = []

# Loop through all chapters (assuming there are 40 chapters)
for chapter_number in range(1, 41):
    chapter_url = f"{base_url}{chapter_number}"
    driver.get(chapter_url)
    
    # Wait for JavaScript to load
    time.sleep(5)
    
    # Get the rendered HTML
    html_content = driver.page_source
    
    # Store the raw HTML content for offline parsing
    chapters_content.append({
        'chapter': f"Chapter {chapter_number}",
        'content': html_content
    })

# Quit the driver
driver.quit()

# Save the raw HTML content to a JSON file
with open('detroit_become_human_chapters_content.json', 'w') as f:
    json.dump(chapters_content, f, indent=4)

print("HTML content has been successfully fetched and saved for offline parsing.")
