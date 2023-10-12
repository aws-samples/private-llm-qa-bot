#!/bin/bash
function usage {
  echo "Hugging Face token is required for downloading the model. Usage: $0 -t TOKEN"
  echo "  -t TOKEN            Hugging Face token (required)"
  echo "  -m MODEL            Model Type (LLM/Embedding, required)"
  exit 1
}

# Parse command-line options
while getopts ":t:m:c:s:" opt; do
  case $opt in
    t) hf_token="$OPTARG" ;;
    m) model_type="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Validate the hf_token and python interpreter exist
if [ -z "$hf_token" ] || [ -z "$model_type" ] || ! command -v python &> /dev/null; then
  usage
fi

eval "$(conda shell.bash hook)"
source /opt/conda/etc/profile.d/conda.sh

conda activate pytorch
pip install -r /home/ubuntu/flask_app/infer/requirements.txt

# If model_type is LLM, download LLM model
if [ "$model_type" = "LLM" ]
then
    # Download models from huggingface
    cd /home/ubuntu/flask_app/models/instruct
    bash prepare_model.sh -t $hf_token

    # Start App
    cd /home/ubuntu/flask_app
    python instruct_app.py --host 0.0.0.0 --port 3000
elif [ "model_type" = "Embedding" ]
then 
    # Download models from huggingface
    cd /home/ubuntu/flask_app/models/embedding
    bash prepare_model.sh -t $hf_token

    cd /home/ubuntu/flask_app/models/cross
    bash prepare_model.sh -t $hf_token

    # Start App
    cd /home/ubuntu/flask_app
    python embedding_app.py --host 0.0.0.0 --port 3000
else
    echo "Invalid model type"
fi