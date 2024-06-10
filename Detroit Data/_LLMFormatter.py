import json

# Define which speakers are considered as "user"
user_speakers = {"Connor", "Kara", "Markus"}

# Function to create formatted data for fine-tuning
def create_fine_tuning_data(parsed_data):
    fine_tuning_data = []

    for chapter in parsed_data:
        chapter_title = chapter['chapter']
        messages = []

        for item in chapter['items']:
            if item['type'] == 'dialogue':
                speaker = item['speaker']
                text = item['text']
                role = "user" if speaker in user_speakers else "assistant"
                messages.append({"role": role, "content": f"{speaker}: {text}"})
            elif item['type'] == 'action':
                action_text = f"*{item['content']}*"
                # Add the action as a separate user message
                messages.append({"role": "user", "content": action_text})
            elif item['type'] == 'choice_group':
                for choice in item['choices']:
                    for dialogue in choice['choice_dialogues']:
                        speaker = dialogue['speaker']
                        text = dialogue['text']
                        role = "user" if speaker in user_speakers else "assistant"
                        messages.append({"role": role, "content": f"{speaker}: {text}"})

        if messages:
            fine_tuning_data.append({"chapter": chapter_title, "messages": messages})

    return fine_tuning_data

# Load the parsed data from the JSON file
with open('detroit_become_human_parsed_data.json', 'r', encoding='utf-8') as f:
    parsed_data = json.load(f)

# Create fine-tuning data
fine_tuning_data = create_fine_tuning_data(parsed_data)

# Save the fine-tuning data to a new JSON file
with open('detroit_become_human_fine_tuning_data.json', 'w', encoding='utf-8') as f:
    json.dump(fine_tuning_data, f, indent=4, ensure_ascii=False)

print("Fine-tuning data has been successfully created and saved.")
