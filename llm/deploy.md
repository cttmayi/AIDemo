# 环境

使用 ollama

选择 qwen， 细节查看 https://ollama.com/library/qwen2

运行命令 ollama run qwen2:0.5b， 下载

测试 http://127.0.0.1:11434/，




# llm_cfg 配置

    llm_cfg = {
        'model': 'qwen2:7b', 
        'model_server': 'http://localhost:11434/v1/',
    }

# 开源大模型safetensors格式转gguf

https://blog.csdn.net/weixin_46248339/article/details/139502733