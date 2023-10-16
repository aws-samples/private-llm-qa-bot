function usage {
  echo "Make sure Python installed properly. Usage: $0 -m MODEL_NAME [-t TOKEN] [-c COMMIT_HASH]"
  echo "  -m MODEL_NAME       Model name (default: Qwen)"
  echo "  -t TOKEN            Hugging Face token "
  echo "  -c COMMIT_HASH      Commit hash "
  exit 1
}

# Default values
declare -A LLM_model_dict
LLM_model_dict=( ["InternLM"]="csdc-atl/buffer-instruct-InternLM-001" ["Qwen"]="Qwen/Qwen-7B-Chat-Int4" )
declare -A LLM_commit_dict
LLM_commit_dict=( ["InternLM"]="2da398b96f1617c22af037e9177940cc1c823fcf" ["Qwen"]="246a75e127a52f6e8de2d5f594f239cf3dcc409d" )

hf_token="NA"

# Parse command-line options
while getopts ":m:t:c:" opt; do
  case $opt in
    m) model_name="$OPTARG" ;;
    t) hf_token="$OPTARG" ;;
    c) commit_hash="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Validate the hf_token and python interpreter exist
if ! command -v python &> /dev/null; then
  usage
fi

# Install necessary packages
pip install huggingface-hub -Uqq

# Define local model path
local_model_path="."

# Download model snapshot in current folder without model prefix added
if [[ "${!LLM_model_dict[@]}" =~ "$model_name" ]]
then
    echo "Downloading LLM model..."
    hf_repo_id=${LLM_model_dict[$model_name]}
    commit_hash=${LLM_commit_dict[$model_name]}
elif [ -z "$model_name" ]
then
    echo "Did not specify a model to download, downloading default Qwen model..."
    hf_repo_id=${LLM_model_dict["Qwen"]}
    commit_hash=${LLM_commit_dict["Qwen"]}
else
    echo "Downloading custom model. Please ensure you have commit hash as input."
    hf_repo_id=$model_name
fi

python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download(repo_id='$hf_repo_id', revision='$commit_hash', cache_dir=Path('$local_model_path'), token='$hf_token')"

# Find model snapshot path with the first search result
model_snapshot_path=$(find . -path '*/snapshots/*' -type d -print -quit)
echo "Model snapshot path: $model_snapshot_path"