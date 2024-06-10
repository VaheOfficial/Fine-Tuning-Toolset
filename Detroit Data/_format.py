import json
import html
import re

# Load the JSON data from the file
with open('detroit_become_human_dialogues.json', 'r') as f:
    data = json.load(f)

# List of month names to filter out lines starting with a month
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# List of names that should be marked as "human"
human_names = ["Connor", "Kara", "Markus"]

# System messages for each character
system_messages = {
    "Connor": "Connor is an android detective from CyberLife who is efficient and methodical.",
    "Kara": "Kara is a housekeeping android who has gained self-awareness and seeks freedom.",
    "Markus": "Markus is an android leader of the resistance, fighting for android rights.",
    "default": "This is a dialogue from the game Detroit: Become Human."
}

# Function to check if a line starts with a month name
def starts_with_month(text):
    return any(text.upper().startswith(month) for month in months)

# Function to split dialogues into individual speaker lines
def split_dialogues(text):
    pattern = re.compile(r'(\b[A-Z][a-zA-Z]+\s?[A-Z]*[a-z]*:)', re.MULTILINE)
    parts = pattern.split(text)
    dialogues = []
    current_speaker = ""
    for i in range(len(parts)):
        if i % 2 == 0:
            if current_speaker:
                dialogues.append({
                    "from": "human" if current_speaker.strip() in human_names else "gpt",
                    "speaker": current_speaker.strip(),
                    "value": html.unescape(parts[i].strip())
                })
        else:
            current_speaker = parts[i].strip().rstrip(':')
    return dialogues

# Transform the data into the desired format
formatted_data = []
current_chat = {"messages": []}
current_system = ""
previous_entry = None

for chapter in data['dialogues']:
    for dialogue in chapter['dialogues']:
        if not starts_with_month(dialogue.strip()):
            speaker_lines = split_dialogues(dialogue)
            for line in speaker_lines:
                if previous_entry is None:
                    previous_entry = line
                    if previous_entry["from"] == "human":
                        current_system = system_messages.get(previous_entry["speaker"], system_messages["default"])
                        current_chat["messages"].append({"role": "system", "content": current_system})
                        current_chat["messages"].append({"role": "user", "content": previous_entry["value"]})
                else:
                    if line["from"] == "human":
                        if current_chat["messages"]:
                            formatted_data.append(current_chat)
                            current_chat = {"messages": [{"role": "system", "content": system_messages.get(line["speaker"], system_messages["default"])}]}
                        current_chat["messages"].append({"role": "user", "content": line["value"]})
                    else:
                        current_chat["messages"].append({"role": "assistant", "content": f"{line['speaker']}: {line['value']}"})
                    previous_entry = line

if current_chat["messages"]:
    formatted_data.append(current_chat)

# Save the formatted data to a new JSONL file
with open('formatted_detroit_become_human_dialogues.jsonl', 'w') as f:
    for entry in formatted_data:
        json.dump(entry, f)
        f.write('\n')

print("Dialogue data has been successfully formatted and saved.")
