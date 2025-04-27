import json
import jsonlines

def convert_to_grpo_format(input_file, output_file):
    # Read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert and write to new format
    with jsonlines.open(output_file, 'w') as writer:
        for item in data:
            # Clean question format
            question = item['question'].strip('"""').strip()
            
            # Construct new data format
            new_item = {
                'messages': [
                    {
                        'role': 'user',
                        'content': f'{question}'
                    }
                ],
                'solution': item['answer']
            }
            
            writer.write(new_item)

if __name__ == '__main__':
    convert_to_grpo_format('results/inter/train_data.json', 'results/inter/train_data_grpo.jsonl')
    convert_to_grpo_format('results/inter/val_data.json', 'results/inter/val_data_grpo.jsonl')
    convert_to_grpo_format('results/inter/test_data.json', 'results/inter/test_data_grpo.jsonl')