'''
llm训练的成本计算
算力（FLOPS）C = 6ND（N为模型参数，D为训练的tokens数）公式计算
训练时间=(8*tokens*model_parameter)/gpu个数*GPU峰值算力（flops）*gpy利用率（0.3-0.55）
'''
import json
import loguru
class EstimateTrainLLMGPU():
    def __init__(self,model_name:str="phi-4",gpu_num=1,gpu_type="H100_SXM",train_tokens=9.8):
        self.tensorType={
            "float32":4,
            "bfloat16":2,
            "float16":2,
            "int4":1
            }
        self.gpu_num = gpu_num
        self.gpu_use_coefficient=0.4
        self.gpu_type = gpu_type
        self.train_tokens=train_tokens
        self.model_name = model_name
        self.model_config:dict = self._read_llm_config()[model_name]
        self.gpu_config:dict = self._read_gpu_config()[gpu_type]
    def _read_llm_config(self):
        with open("llm_config.json","r",encoding="utf-8") as file:
            model_config = json.loads(file.read())
        return model_config
    
    def _read_gpu_config(self):
        with open("gpu_config.json","r",encoding="utf-8") as file:
            gpu_config = json.loads(file.read())
        return gpu_config

    def calculate_llm_paremeter(self):
        '''
        根据配置文件计算出参数，根据参数和数据类型计算显存大小,单位是GB
        '''
        model_paremeter = (self.model_config["num_hidden_layers"]*(12*pow(self.model_config['hidden_size'],2)+13*self.model_config['hidden_size']) +self.model_config["vocab_size"]*self.model_config['hidden_size'])
        return model_paremeter
    
    
    def calculate_llm_train_time(self):
        train_time = (8*self.train_tokens*self.calculate_llm_paremeter()*pow(10,12))/(self.gpu_num*self.gpu_config['peak_bfp16_tflops']*self.gpu_use_coefficient*pow(10,12))
        train_time_day = train_time/3600/24
        return train_time_day
    
    def calculate_llm_train_gpu_memory(self):
        pass
    
    def select_gpu_type_train(self):
        pass
    
    def calculate_llm_train_lora_gpu_memory(self):
        pass
    
    def calculate_llm_train_lora_time(self):
        pass
    
if __name__ == "__main__":
    loguru.logger.info(f"estimate llm train gpu memeory or time")
    model = EstimateTrainLLMGPU()
    train_hour = model.calculate_llm_train_time()
    loguru.logger.info(f"result:{train_hour}")
    
    