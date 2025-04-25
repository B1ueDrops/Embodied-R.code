# Swift-based Consistency Verification Model Service

## Overview

This service uses the ms-swift framework to implement consistency verification functionality by running a model service on the fifth GPU (GPU 4), avoiding interference with the training process on the first four GPUs. The service is provided via an API, making consistency verification more reliable and stable.

## File Description

- `start_consistency_model_server.py`: The main program of the Swift-based consistency verification model API service, running on GPU 4
- `start_consistency_service.sh`: The script to start the Swift-based consistency verification model service
- `test_consistency_service.py`: The script to test the Swift-based consistency verification model service
- `train_5GPUs.sh`: The training script that integrates the Swift-based consistency verification model service

## Installation and Configuration

### 1. Configuration

Default configuration:

- Model name: `Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`
- API service port: 8000
- GPU: 4th GPU

If you need to modify the configuration, please edit the following files:

- `start_consistency_model_server.py`: Modify model name and port
- `start_consistency_service.sh`: Modify GPU number

### 2. Supported Models

The following models are supported:

1. **Qwen2.5-VL-3B-Instruct**

   - Model name: `Qwen/Qwen2.5-VL-3B-Instruct`
2. **Qwen2.5-3B-Instruct**

   - Model name: `Qwen/Qwen2.5-3B-Instruct`

## Usage

### 1. Start the Service

```bash
bash train/start_consistency_service.sh
```

After the service starts, it will run in the background, and the logs will be saved in `train/consistency_service.log`.

### 2. Test the Service

```bash
python train/test_consistency_service.py
```

If the service is running normally, it will display a message indicating that the test passed.

### 3. Integrate with Training

Use the modified training script, which has already integrated the process of starting the Swift-based consistency verification model service:

```bash
bash train/train_5GPUs.sh
```

## API Interface Description

### Health Check

- URL: `http://localhost:8000/health`
- Method: GET
- Response: `{"status": "healthy", "model_loaded": true}`

### Chat Completion

- URL: `http://localhost:8000/v1/chat/completions`
- Method: POST
- Request body:
  ```json
  {
    "model": "qwen2.5-vl-3b-instruct",
    "messages": [
      {"role": "system", "content": "system prompt"},
      {"role": "user", "content": "user prompt"}
    ],
    "temperature": 0.1,
    "max_tokens": 10
  }
  ```
- Response:
  ```json
  {
    "id": "consistency-model-response",
    "object": "chat.completion",
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "assistant response"
        },
        "index": 0
      }
    ]
  }
  ```

## Working Principle

1. When the service starts, the ms-swift framework loads the model on GPU 4 (default Qwen2.5-VL-3B-Instruct, fallback Qwen2.5-3B-Instruct if loading fails)
2. The training script runs on the first four GPUs (0-3)
3. When consistency verification is needed, `consistency_reward_local.py` makes HTTP requests to the local API service
4. The API service receives requests, generates responses using the loaded model, and returns the results
5. `consistency_reward_local.py` compares the model's response with the training model's response to calculate the reward

## How to Close the Service

There are multiple methods to close the Swift-based consistency verification model service:

1. **Use pkill command** (recommended):

   ```bash
   pkill -f "start_consistency_model_server.py"
   ```
2. **Find and terminate the process**:

   ```bash
   # Find service process ID
   ps aux | grep "start_consistency_model_server.py"

   # Terminate process (replace <PID> with actual process ID)
   kill -9 <PID>
   ```
3. **Check if the service has been closed**:

   ```bash
   # Check if port 8000 is still occupied
   netstat -tuln | grep 8000

   # Try to access the health check endpoint
   curl http://localhost:8000/health
   ```

## Troubleshooting

1. **If the service fails to start, check**:

   - Whether GPU 4 is available
   - Whether the model path is correct
   - Whether the dependencies are installed successfully
2. **If API calls fail, check**:

   - Whether the service is running (using `ps aux | grep start_consistency_model_server.py`)
   - Whether the port is occupied (using `netstat -tuln | grep 8000`)
   - Whether there are error messages in the log (view `train/consistency_service.log`)
3. **If Qwen2.5-VL model loading fails**:

   - The system will automatically try to use the backup Qwen2.5-3B-Instruct model
   - View the log file to understand the detailed error information
