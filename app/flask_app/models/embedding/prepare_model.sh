function usage {
  echo "Make sure Python installed properly. Usage: $0 -m MODEL_NAME [-t TOKEN] [-c COMMIT_HASH]"
  echo "  -m MODEL_NAME       Model name (default: BGE)"
  echo "  -t TOKEN            Hugging Face token "
  echo "  -c COMMIT_HASH      Commit hash"
  exit 1
}

# Default values
declare -A Embedding_model_dict
Embedding_model_dict=( ["BGE"]="BAAI/bge-large-zh-v1.5" )
declare -A Embedding_commit_dict
Embedding_commit_dict=( ["BGE"]="00f8ffc4928a685117583e2a38af8ebb65dcec2c" )

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
if [[ "${!Embedding_model_dict[@]}" =~ "$model_name" ]]
then
    echo "Downloading Embedding model..."
    hf_repo_id=${Embedding_model_dict[$model_name]}
    commit_hash=${Embedding_commit_dict[$model_name]}
elif [ -z "$model_name" ]
then
    echo "Did not specify a model to download, downloading default BGE model..."
    hf_repo_id=${Embedding_model_dict["BGE"]}
    commit_hash=${Embedding_commit_dict["BGE"]}
else
    echo "Downloading custom model. Please ensure you have commit hash as input."
    hf_repo_id=$model_name
fi

python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download(repo_id='$hf_repo_id', revision='$commit_hash', cache_dir=Path('$local_model_path'), token='$hf_token')"

# Find model snapshot path with the first search result
model_snapshot_path=$(find . -path '*/snapshots/*' -type d -print -quit)
echo "Model snapshot path: $model_snapshot_path"