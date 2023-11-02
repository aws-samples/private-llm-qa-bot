# 使用方法



- 测试Ask_Assistant lambda (该lambda为chatbot的主入口), llm_model_name 可以是'claude'和'claude-instant'
    ```
    python ask_assistant_benchmark.py --repeat_num=100 --llm_model_name='claude' --embedding_model_endpoint='{embedding_model_endpoint}'
    ```

- 测试Detect_Intention lambda, llm_model_name 可以是'claude'和'claude-instant'
    ```
    python intention_detect_benchmark.py --repeat_num=1 --llm_model_name='claude' 
    ```