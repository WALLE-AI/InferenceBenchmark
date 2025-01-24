##https://github.com/vllm-project/vllm/blob/f35ec461fc655a50abc5146fa27a79fdf42f55a1/benchmarks/backend_request_func.py#L37 直接借鉴这里的
#https://tinkerd.net/blog/machine-learning/distributed-training/
'''
可以结合商用API的TTFT TPOT和吞吐量
ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}
'''