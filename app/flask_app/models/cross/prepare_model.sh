function usage {
  echo "Make sure Python installed properly. Usage: $0 -m MODEL_NAME [-t TOKEN] [-c COMMIT_HASH]"
  echo "  -m MODEL_NAME       Model name (default: BufferCross)"
  echo "  -t TOKEN            Hugging Face token "
  echo "  -c COMMIT_HASH      Commit hash "
  exit 1
}

# Default values
declare -A Cross_model_dict
Cross_model_dict=( ["Cross"]="csdc-atl/buffer-cross-001" )
declare -A Cross_commit_dict
Cross_commit_dict=( ["Cross"]="46d270928463db49b317e5ea469a8ac8152f4a13" )

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
if [[ "${!Cross_model_dict[@]}" =~ "$model_name" ]]
then
    echo "Downloading Cross model..."
    hf_repo_id=${Cross_model_dict[$model_name]}
    commit_hash=${Cross_commit_dict[$model_name]}
elif [ -z "$model_name" ]
then
    echo "Did not specify a model to download, downloading default Cross model..."
    hf_repo_id=${Cross_model_dict["Cross"]}
    commit_hash=${Cross_commit_dict["Cross"]}
else
    echo "Downloading custom model. Please ensure you have commit hash as input."
    hf_repo_id=$model_name
fi

python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download(repo_id='$hf_repo_id', revision='$commit_hash', cache_dir=Path('$local_model_path'), token='$hf_token')"

# Find model snapshot path with the first search result
model_snapshot_path=$(find . -path '*/snapshots/*' -type d -print -quit)
echo "Model snapshot path: $model_snapshot_path"