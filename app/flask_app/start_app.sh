#!/bin/bash
function usage {
  echo "Hugging Face token is required for downloading the model. Usage: $0 -m MODEL"
  echo "  -m MODEL            Model Type (InternLM/Qwen/BGE/BGECross required)"
  echo "  -t TOKEN            Hugging Face token"
  echo "  -p PORT             Port for the App(default 3000)"
  exit 1
}

port=3000
hf_token="NA"
deploy_region="global_deploy"

# Parse command-line options
while getopts ":t:m:p:" opt; do
  case $opt in
    t) hf_token="$OPTARG" ;;
    m) model_type="$OPTARG" ;;
    p) port="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Validate the hf_token and python interpreter exist
if [ -z "$model_type" ] || ! command -v python &> /dev/null; then
  usage
fi

eval "$(conda shell.bash hook)"
source /opt/conda/etc/profile.d/conda.sh

conda activate pytorch
pip install -r /home/ubuntu/flask_app/infer/requirements.txt

# If model_type is LLM, download LLM model
if [ "$model_type" = "InternLM" ] || [ "$model_type" = "Qwen" ]
then
    # Download models from huggingface
    cd /home/ubuntu/flask_app/models/instruct
    bash prepare_model.sh -t $hf_token -m $model_type

    # Start App
    cd /home/ubuntu/flask_app
    # python instruct_app.py --model $model_type --host 0.0.0.0 --port $port --deploy_region $deploy_region
    pm2 start instruct_app.py --name llm-app --interpreter python -- --model $model_type --host 0.0.0.0 --port $port --deploy_region $deploy_region
    pm2 startup systemd
elif [ "$model_type" = "BGECross" ]
then 
    # Download models from huggingface
    cd /home/ubuntu/flask_app/models/embedding
    bash prepare_model.sh -t $hf_token -m BGE

    cd /home/ubuntu/flask_app/models/cross
    bash prepare_model.sh -t $hf_token -m Cross

    # Start App
    cd /home/ubuntu/flask_app
    # python embedding_app.py --host 0.0.0.0 --port $port --deploy_region $deploy_region --cross True
    pm2 start embedding_app.py --name embedding-app --interpreter python -- --host 0.0.0.0 --port $port --cross True --deploy_region $deploy_region
    pm2 startup systemd
elif [ "$model_type" = "BGE" ]
then 
    # Download models from huggingface
    cd /home/ubuntu/flask_app/models/embedding 
    bash prepare_model.sh -t $hf_token -m $model_type

    # Start App
    cd /home/ubuntu/flask_app
    # python embedding_app.py --host 0.0.0.0 --port $port --deploy_region $deploy_region
    pm2 start embedding_app.py --name embedding-app --interpreter python -- --host 0.0.0.0 --port $port --deploy_region $deploy_region
    pm2 startup systemd
else
    echo "Invalid model type"
fi