import os
os.environ['VIDEO_MAX_PIXELS'] = '117600'  # Adjust resolution to 420*280
os.environ['VIDEO_TOTAL_PIXELS'] = '3763200'  # Adjust total pixel count
os.environ['FPS'] = '1.0'  # Reduce frame rate
os.environ['FPS_MIN_FRAMES'] = '4'  # Minimum number of frames
os.environ['FPS_MAX_FRAMES'] = '32'  # Reduce maximum number of frames
import json
import argparse
from tqdm import tqdm
from swift.llm import PtEngine, RequestConfig, InferRequest, AdapterRequest, get_template, BaseArguments
from swift.tuners import Swift

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def batch_inference(model_name, json_file, output_file, adapter_path=None, batch_size=2, max_tokens=3096):
    # Load data
    data = load_json(json_file)
    
    # System prompt
    system = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \n<answer> answer here </answer>. Ensure that your answer is consistent with and directly derived from your thinking process, maintaining logical coherence between the two sections. User: . Assistant:'''
    
    # Initialize inference engine
    if adapter_path:
        # Initialize engine with LoRA adapter
        print(f"Using LoRA adapter from: {adapter_path}")
        # Method 1: Directly use adapter path
        engine = PtEngine(model_name, max_batch_size=batch_size, adapters=[adapter_path], default_system=system)
    else:
        # Use original model
        engine = PtEngine(model_name, max_batch_size=batch_size, default_system=system)
    request_config = RequestConfig(max_tokens=max_tokens, temperature=0)
    
    # Process data in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        infer_requests = []
        
        for item in batch:
            question = item['question']
            videos = item.get('videos', [])
            
            # Create inference request
            infer_request = InferRequest(
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': question}
                ],
                videos=videos if videos else None
            )
            infer_requests.append(infer_request)
        
        # Execute batch inference
        print(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        resp_list = engine.infer(infer_requests, request_config)
        
        # Update data
        for j, resp in enumerate(resp_list):
            if i + j < len(data):
                data[i + j]['content'] = resp.choices[0].message.content
    
    # Save results
    save_json(data, output_file)
    print(f"Inference completed. Results saved to {output_file}")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for video question answering")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--input_file", type=str, default=None, help="Input JSON file")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=3096, help="Maximum tokens for generation")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter for inference")
    args = parser.parse_args()
    batch_inference(args.model, args.input_file, args.output_file, args.adapter_path, args.batch_size, args.max_tokens)
