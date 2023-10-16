function usage {
  echo "Make sure Python installed properly. Usage: $0 -m MODEL_NAME [-t TOKEN] [-c COMMIT_HASH]"
  echo "  -m MODEL_NAME       Model name (default: Qwen)"
  echo "  -t TOKEN            ModelScope token "
  echo "  -c COMMIT_HASH      Commit hash "
  exit 1
}

# Default values
declare -A LLM_model_dict
LLM_model_dict=( ["Qwen"]="qwen/Qwen-7B-Chat-Int4" )
declare -A LLM_commit_dict
LLM_commit_dict=( ["Qwen"]="v1.1.4" )

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

# Download model snapshot in current folder without model prefix added
if [[ "${!LLM_model_dict[@]}" =~ "$model_name" ]]
then
    echo "Downloading LLM model..."
    ms_repo_id=${LLM_model_dict[$model_name]}
    commit_hash=${LLM_commit_dict[$model_name]}
elif [ -z "$model_name" ]
then
    echo "Did not specify a model to download, downloading default Qwen model..."
    ms_repo_id=${LLM_model_dict["Qwen"]}
    commit_hash=${LLM_commit_dict["Qwen"]}
else
    echo "Downloading custom model. Please ensure you have commit hash as input."
    ms_repo_id=$model_name
fi

python -c "from modelscope.hub.snapshot_download import snapshot_download; from pathlib import Path; snapshot_download('$ms_repo_id', revision='$commit_hash', cache_dir=Path('$local_model_path'))"
