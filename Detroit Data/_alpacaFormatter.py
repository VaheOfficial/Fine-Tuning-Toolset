import json

# Define which speakers are considered as "user"
user_speakers = {"Connor", "Kara", "Markus"}

# Function to create Alpaca fine-tuning data
def create_alpaca_fine_tuning_data(parsed_data):
    alpaca_fine_tuning_data = []

    for chapter in parsed_data:
        chapter_title = chapter['chapter']

        current_instruction = ""
        current_output = ""
        for item in chapter['items']:
            if item['type'] == 'dialogue':
                speaker = item['speaker']
                text = item['text']
                if speaker in user_speakers:
                    # If there's an existing instruction and output, save them
                    if current_instruction and current_output:
                        alpaca_fine_tuning_data.append({
                            "instruction": current_instruction,
                            "output": current_output
                        })
                        current_output = ""  # Reset the current output

                    current_instruction = f"{speaker}: {text}"
                else:
                    current_output += f"{speaker}: {text}\n"

            elif item['type'] == 'action':
                action_text = f"*{item['content']}*"
                if current_output:
                    current_output += f"{action_text}\n"
                else:
                    current_instruction += f"\n{action_text}"

            elif item['type'] == 'choice_group':
                for choice in item['choices']:
                    for dialogue in choice['choice_dialogues']:
                        speaker = dialogue['speaker']
                        text = dialogue['text']
                        if speaker in user_speakers:
                            if current_instruction and current_output:
                                alpaca_fine_tuning_data.append({
                                    "instruction": current_instruction,
                                    "output": current_output
                                })
                                current_output = ""  # Reset the current output

                            current_instruction = f"{speaker}: {text}"
                        else:
                            current_output += f"{speaker}: {text}\n"

        # Append any remaining instruction and output
        if current_instruction and current_output:
            alpaca_fine_tuning_data.append({
                "instruction": current_instruction,
                "output": current_output
            })

    return alpaca_fine_tuning_data

# Load the parsed data from the JSON file
with open('detroit_become_human_parsed_data.json', 'r') as f:
    parsed_data = json.load(f)

# Create Alpaca fine-tuning data
alpaca_fine_tuning_data = create_alpaca_fine_tuning_data(parsed_data)

# Save the Alpaca fine-tuning data to a new JSON file
with open('detroit_become_human_alpaca_fine_tuning_data.json', 'w') as f:
    json.dump(alpaca_fine_tuning_data, f, indent=4, ensure_ascii=True)

print("Alpaca fine-tuning data has been successfully created and saved.")
