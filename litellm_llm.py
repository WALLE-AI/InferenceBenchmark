import base64
import json
import threading
import time
import uuid
import concurrent
from litellm import acompletion, completion
import asyncio
import random
from PIL import Image
from io import BytesIO
import tiktoken

import loguru


tiktonken_encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(tiktonken_encoder.encode(string))
    except Exception:
        return 0


def get_ttft():
    '''
    获取首个字符出现的时间
    '''
    pass
def get_ttop():
    '''
    获取decode token的平均间隔时间
    
    '''
    pass
def get_throughput():
    '''
    获取所有all tokens
    '''
    pass




class MeasureExecutionTime:
    """
    装饰器类，用于测量另一个函数的执行时间，并统计总时间
    """
    total_time = 0 

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()  # 开始时间
        result = self.func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        execution_time = end_time - start_time  # 计算执行时间
        self.total_time += execution_time  # 累加到总时间
        loguru.logger.info(f"function {self.func.__name__} execute time：{execution_time:.6f} 秒")
        loguru.logger.info(f"function {self.func.__name__} total execute time：{self.total_time:.6f} 秒")
        return result
    @classmethod
    def get_total_time(cls):
        """sssssssss
        返回函数调用的总执行时间
        """
        return cls.total_time
    
def single_measure_execution_time(func):
    """
    装饰器函数，用于测量另一个函数的执行时间,调用一次时间计算
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        execution_time = end_time - start_time  # 计算执行时间
        loguru.logger.info(f"function {func.__name__} execute time：{execution_time:.6f} 秒")
        return result,execution_time
    return wrapper


# image_path = "datasets/images/dda5077ff17c4c099d103d8453448a46.png"
image_path = "test_data/images/aee000a0-75b9-2c97-437d-72d658d25b24.jpg"

def image_to_base64(image_path):

    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        mime_type = image_path.split(".")[-1]
        with Image.open(image_path) as img:
            # 定义新的尺寸，例如缩小到原来的一半
            new_width = img.width // 2
            new_height = img.height // 2
            # 调整图片大小
            img_resized = img.resize((new_width, new_height))
            # 将图片转换为字节流
            buffered = BytesIO()
            img_resized.save(buffered, format=img.format)
            # 将字节流转换为Base64编码
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'data:image/{mime_type};base64,{img_base64}'
    
all_texts = []
all_tokens_list = []

def build_image_prompt():
        user_content = [
                {"type": "text",
                 "text": "详细描述该图片中的内容"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url":image_to_base64(image_path) ,
                    },
                }
        ]
        prompt = [{"role": "user", "content": user_content}]
        return prompt
##model="hosted_vllm/InternVL2-40B", max_retries=1,
async def async_chat_completion(**kwargs):
    semaphore = asyncio.Semaphore(10)
    async with semaphore:
        for attempt in range(kwargs.get("max_retries")):
            try:
                messages = build_image_prompt()
                response = await acompletion(
                    model=kwargs.get("model_name"),
                    messages=messages, 
                    stream=True,
                    temperature=0.0,
                    # base_url = "http://172.18.204.2:9992/v1",
                    base_url = kwargs.get("base_url"),
                    api_key="emty",
                )
                texts = ""
                async for text in response:
                    content = text["choices"][0].delta.content
                    if text["choices"][0].finish_reason != "stop":
                        texts += content
                    # if text["usage"]:
                    #     usage_info_dict = text.usage.to_dict()
                    #     loguru.logger.info(f"tokens:{usage_info_dict}")
                data = {
                    "id": str(uuid.uuid4()),
                    "text":texts,
                }
                all_tokens = num_tokens_from_string(texts)
                all_tokens_list.append(all_tokens)
                all_texts.append(data)
                return texts.strip()
            except Exception as e:
                base_wait = 2**attempt
                jitter = random.uniform(0, base_wait * 0.1)  # 10% jitter
                wait_time = base_wait + jitter
                print(f"Error during API call: {e}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        print("Max retries exceeded.")
        return None
    
    
def chat_completion(**kwargs):
    try:
        messages = build_image_prompt()
        response = completion(
            model=kwargs.get("model_name"),
            messages=messages, 
            stream=True,
            temperature=0.0,
            base_url = kwargs.get("base_url"),
            api_key="emty",
        )
        texts = ""
        for text in response:
            content = text["choices"][0].delta.content
            if text["choices"][0].finish_reason != "stop":
                texts += content
            # if text["usage"]:
            #     usage_info_dict = text.usage.to_dict()
            #     loguru.logger.info(f"tokens:{usage_info_dict}")
        data = {
            "id": str(uuid.uuid4()),
            "text":texts,
        }
        all_tokens = num_tokens_from_string(texts)
        all_tokens_list.append(all_tokens)
        all_texts.append(data)
        return texts.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
    
    
@single_measure_execution_time
def execute_no_thread():
    for index in range(10):
        asyncio.run(async_chat_completion(max_retries=2,model_name="hosted_vllm/InternVL2-40B"))
        
    
    
def semaphore_do_work(semaphore, thread_name,**kwargs):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        asyncio.run(async_chat_completion(**kwargs))
        loguru.logger.info(f"{thread_name} is done")
        
def semaphore_do_work_no_async(semaphore, thread_name,**kwargs):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        chat_completion(**kwargs)
        loguru.logger.info(f"{thread_name} is done")


@single_measure_execution_time  
def execute_threading_max(exe_func,base_url:str,max_thread_num=16):
    max_threads = max_thread_num
    semaphore = threading.Semaphore(max_threads)
    threads=[]
    thread_name = 0
    for index in range(100):
        document_format_thread = threading.Thread(
                        target=exe_func,
                        kwargs={
                            "semaphore":semaphore,
                            "thread_name":thread_name,
                            "max_retries":  2,
                            "model_name":"hosted_vllm/starvlm-qwen2_vl-7b",
                            "base_url":base_url
                        }
                    )
                ##执行线程
        thread_name+=1
        threads.append(document_format_thread)
        document_format_thread.start()
    for thread in threads:
        thread.join()
    with open("test.jsonl","w",encoding="utf-8") as file:
        loguru.logger.info(f"all texts:{len(all_texts)}")
        file.write(json.dumps(all_texts,ensure_ascii=False,indent=4))
        
def run_async_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_chat_completion())
    
def execute_task():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_async_in_thread) for _ in range(10)]

    # 等待所有线程完成
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(result)
    
if __name__ == "__main__":
    loguru.logger.info(f"test litellm")
    import numpy as np
    # execute_threading_max(exe_func=semaphore_do_work)
    base_urls = ["http://xxxx:9992/v1"]
    for base_url in base_urls:
        execute_threading_max(exe_func=semaphore_do_work_no_async,base_url=base_url)
    loguru.logger.info(f"all tokens :{np.sum(all_tokens_list)}")
    
    