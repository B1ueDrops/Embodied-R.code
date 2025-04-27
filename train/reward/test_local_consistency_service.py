import requests
import sys
import time
import json

import requests
import sys
import time
import json

def test_api_service(base_url="http://localhost:8000"):
    """Test if the API service is working properly"""
    print(f"Testing Swift-based API service: {base_url}")
    
    # Test health check endpoint
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("Health check passed")
        else:
            print(f"Health check failed, status code: {health_response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Health check request failed: {str(e)}")
        print("Please ensure the service is running, execute: bash train/reward/start_consistency_service.sh")
        return False
    
    # Test inference endpoint
    try:
        # Build a simple multiple-choice test
        test_request = {
            "model": "qwen2.5-3b-instruct",
            "messages": [
                {"role": "system", "content": "You are a professional question answering assistant."},
                {"role": "user", "content": "This is a multiple-choice question. Please select the most appropriate option (a letter from A-D) based on the following question and thinking process.\n\nYou only need to answer with one letter, no explanation or other content.\n\nQuestion: Xiao Ming has 5 apples, Xiao Hong has 3 apples, Xiao Gang has 7 apples. Who has the most apples?\n\nThinking process: Xiao Ming has 5 apples, Xiao Hong has 3 apples, Xiao Gang has 7 apples. Comparing the three people's apple quantities: 5, 3, 7, of which 7 is the largest. So Xiao Gang has the most apples.\n\nPlease answer with only one letter (A-D): A. Xiao Ming B. Xiao Hong C. Xiao Gang D. All three have the same amount"}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        print("Sending test request...")
        start_time = time.time()
        response = requests.post(f"{base_url}/v1/chat/completions", json=test_request, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                print(f"Inference successful, response time: {end_time - start_time:.2f} seconds")
                print(f"Model answer: {answer}")
                
                # Check if the answer contains the correct option C
                if "C" in answer:
                    print("Answer is correct, contains option C (Xiao Gang)")
                else:
                    print("Answer may be incorrect, does not clearly contain option C (Xiao Gang)")
                
                return True
            else:
                print("Inference response format is incorrect")
                return False
        else:
            print(f"Inference request failed, status code: {response.status_code}")
            print(f"Error message: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Inference request exception: {str(e)}")
        return False

if __name__ == "__main__":
    # API service URL can be specified via command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    if test_api_service(base_url):
        print("API service test passed, Swift-based consistency check model service is working properly")
        sys.exit(0)
    else:
        print("API service test failed, please check logs and service status")
        sys.exit(1)
