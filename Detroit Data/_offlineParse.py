import json
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to parse individual chapter content
def parse_chapter(chapter):
    chapter_title = chapter['chapter']
    soup = BeautifulSoup(chapter['content'], 'html.parser')
    # Initialize structures to store parsed items
    items = []
    dialogueItems = []
    # Extract dialogues and actions in chronological order
    for current_tag in soup.find_all(['h1', 'h2', 'app-unlock-condition', 'div', 'app-dialogue-line', 'app-choice-group']):
        if current_tag.name == 'h1':
            items.append({'type': 'section', 'content': current_tag.text.strip()})
        elif current_tag.name == 'h2':
            items.append({'type': 'action', 'content': current_tag.text.strip()})
        elif current_tag.name == 'app-unlock-condition':
            items.append({'type': 'action', 'content': current_tag.text.strip()})
        elif current_tag.name == 'div' and 'condition' in current_tag.get('class', []):
            items.append({'type': 'condition', 'content': current_tag.text.strip()})
        elif current_tag.name == 'app-choice-group':
            choices = []
            for choice_item in current_tag.find_all('app-choice-item'):
                choice_title = choice_item.find('span').text.strip()
                choice_dialogues = []
                for line in choice_item.find_all('app-dialogue-line'):
                    speaker = line.get('name')
                    dialogue_span = line.find('span', class_='text-gray-800 font-bold dark:text-gray-200')
                    if dialogue_span:
                        dialogue_text = dialogue_span.find_next_sibling(string=True)
                        if dialogue_text:
                            dialogueItems.append(dialogue_text.strip())
                            choice_dialogues.append({
                                'speaker': speaker,
                                'text': dialogue_text.strip()
                            })
                choices.append({
                    'type': 'choice',
                    'title': choice_title,
                    'choice_dialogues': choice_dialogues
                })
            items.append({
                'type': 'choice_group',
                'choices': choices
            })
        elif current_tag.name == 'app-dialogue-line':
            speaker = current_tag.get('name')
            dialogue_span = current_tag.find('span', class_='text-gray-800 font-bold dark:text-gray-200')
            if dialogue_span:
                dialogue_text = dialogue_span.find_next_sibling(string=True)
                if dialogue_text and dialogue_text.strip() not in dialogueItems:
                    items.append({
                        'type': 'dialogue',
                        'speaker': speaker,
                        'text': dialogue_text.strip()
                    })

    # Store the collected data for this chapter
    return {
        'chapter': chapter_title,
        'items': items
    }

# Load the saved HTML content from the JSON file
with open('detroit_become_human_chapters_content.json', 'r') as f:
    chapters_content = json.load(f)

# Initialize an empty list to store parsed data
parsed_data = []

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(parse_chapter, chapter) for chapter in chapters_content]
    for future in as_completed(futures):
        parsed_data.append(future.result())

# Save the parsed data to a JSON file
with open('detroit_become_human_parsed_data.json', 'w') as f:
    json.dump(parsed_data, f, indent=4)

print("HTML content has been successfully parsed and saved.")