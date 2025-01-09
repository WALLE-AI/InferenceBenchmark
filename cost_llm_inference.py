'''
统计出模型的需要的GPU资源以及对应硬件的性能
1、GPU硬件型号与内存大小：A100 40G
2、推理框架：vllm
3、初始启动预估的显存大小：参数内存：70*2+2*tensorType*hidde*layer*并发数+Memory Overhead（前面总和*0.1）
4、各个LLM配置参数
参考：https://mp.weixin.qq.com/s/CJxA-PxF_lvSpMr_7uHVVg 但是具体还是根据具体的业务场景进行，这里只是做预估
'''

import json

import loguru
import numpy as np

class EstimateInferenceLLMGPU():
    def __init__(self,model_name:str="Llama_2_13b_chat_hf",concurrent:int=1,need_tokens=4096):
        self.tensorType={
            "float32":4,
            "bfloat16":2,
            "float16":2,
            "int4":1
            }
        self.coefficient=0.1
        self.concurrent=concurrent
        self.need_tokens = need_tokens
        self.model_name = model_name
        self.model_config:dict = self._read_llm_config()[model_name]
        self.gpu_config:dict = self._read_gpu_config()
    def _read_llm_config(self):
        with open("llm_config.json","r",encoding="utf-8") as file:
            model_config = json.loads(file.read())
        return model_config
    
    def _read_gpu_config(self):
        with open("gpu_config.json","r",encoding="utf-8") as file:
            gpu_config = json.loads(file.read())
        return gpu_config
    
    def get_llm_info(self):
        '''
        获取张量类型、隐含层数和隐函数
        '''
        tensortype=self.model_config["torch_dtype"]
        tensortype,hidde_layer,layer_num =self.tensorType[tensortype],self.model_config["num_hidden_layers"],self.model_config['hidden_size']
        return tensortype,hidde_layer,layer_num
    
    def calculate_llm_paremeter_to_memory(self):
        '''
        根据配置文件计算出参数，根据参数和数据类型计算显存大小,单位是GB
        '''
        model_paremeter = (self.model_config["num_hidden_layers"]*(12*pow(self.model_config['hidden_size'],2)+13*self.model_config['hidden_size']) +self.model_config["vocab_size"]*self.model_config['hidden_size'])/pow(10,9)
        return model_paremeter*2
    
    def calculate_llm_kv_to_memory(self):
        '''
        估算出kv显存大小,单位GB
        '''
        tensortype,hidde_layer,layer_num = self.get_llm_info()
        single_token = 2*tensortype*hidde_layer*layer_num
        all_memeory = (single_token*self.concurrent*self.need_tokens)/pow(10,9)
        return all_memeory
    
    def calculate_llm_memeory_overhead_to_memory(self):
        '''
            额外显存大小
        '''
        return self.coefficient*(self.calculate_llm_kv_to_memory()+self.calculate_llm_paremeter_to_memory())
    
    def run_estimate_gpu_memory(self):
        llm_paremeter = self.calculate_llm_paremeter_to_memory()
        kv = self.calculate_llm_kv_to_memory()
        over_head_memeory = self.calculate_llm_memeory_overhead_to_memory()
        return np.sum([llm_paremeter,kv,over_head_memeory])
    
    def select_gpu_type(self):
        select_gpu_result = []
        all_memeory = self.run_estimate_gpu_memory()
        for key,value in self.gpu_config.items():
            gpu_memeory_num = value['gpu_memory']
            if all_memeory < gpu_memeory_num:
                gpu_data = {
                    "gpu_type":key,
                    "gpu_memory":value['gpu_memory'],
                    "gpu_num":1,
                    "llm_need_memory":all_memeory,
                    "model_name":self.model_name,
                    "coefficient":self.coefficient,
                    "concurrent":self.concurrent,
                    "need_tokens":self.need_tokens
                }
                select_gpu_result.append(gpu_data)
        return select_gpu_result
    
if __name__ == "__main__":
    loguru.logger.info(f"estimate llm gpu memeory")
    model = EstimateInferenceLLMGPU()
    result = model.select_gpu_type()
    loguru.logger.info(f"result:{result}")
    
    
    