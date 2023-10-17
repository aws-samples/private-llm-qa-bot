function usage {
  echo "Make sure Python installed properly. Usage: $0 -m MODEL_NAME [-t TOKEN] [-c COMMIT_HASH]"
  echo "  -m MODEL_NAME       Model name (default: BGE)"
  echo "  -t TOKEN            ModelScope token "
  echo "  -c COMMIT_HASH      Commit hash "
  exit 1
}

# Default values
declare -A Embedding_model_dict
Embedding_model_dict=( ["BGE"]="Xorbits/bge-large-zh-v1.5" )
declare -A Embedding_commit_dict
Embedding_commit_dict=( ["BGE"]="master" )

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
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# Define local model path
local_model_path="."

# Download model snapshot in current folder
if [[ "${!Embedding_model_dict[@]}" =~ "$model_name" ]]
then
    echo "Downloading Embedding model..."
    ms_repo_id=${Embedding_model_dict[$model_name]}
    commit_hash=${Embedding_commit_dict[$model_name]}
elif [ -z "$model_name" ]
then
    echo "Did not specify a model to download, downloading default BGE model..."
    ms_repo_id=${Embedding_model_dict["BGE"]}
    commit_hash=${Embedding_commit_dict["BGE"]}
else
    echo "Downloading custom model. Please ensure you have commit hash as input."
    ms_repo_id=$model_name
fi

python -c "from modelscope.hub.snapshot_download import snapshot_download; from pathlib import Path; snapshot_download('$ms_repo_id', revision='$commit_hash', cache_dir=Path('$local_model_path'))"
