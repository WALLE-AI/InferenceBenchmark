import json
import gradio as gr
import loguru

from cost_llm_inference import EstimateInferenceLLMGPU
from cost_llm_train import EstimateTrainLLMGPU


css = """
.tab-label {
    font-size: 48px !important;  /* 设置标签页标题的字体大小 */
}
.tab-content {
    font-size: 36px !important;  /* 设置标签页内容的字体大小 */
}
"""
def get_model_list():
    with open("llm_config.json","r",encoding="utf-8") as file:
        model_config = json.loads(file.read())
        all_model_list = [key for key,value in model_config.items()]
    return all_model_list

def get_gpu_list():
    with open("gpu_config.json","r",encoding="utf-8") as file:
        model_config = json.loads(file.read())
        all_gpu_list = [key for key,value in model_config.items()]
    return all_gpu_list

def llm_gpu_inference_memory_func(model_name,concurrent, gen_need_tokens):
    model = EstimateInferenceLLMGPU(model_name=model_name[0],concurrent=concurrent,need_tokens=gen_need_tokens)
    result = model.select_gpu_type()
    return result

def estimate_llm_train_day(model_name,gpu_num,gpu_type,train_tokens):
    model = EstimateTrainLLMGPU(model_name[0],gpu_num,gpu_type[0],train_tokens)
    train_time_day = model.calculate_llm_train_time()
    return train_time_day
    


with gr.Blocks() as app:
    gr.Markdown("""<center><font size=8>LLM GPU Performance Estimate</center>""")
    with gr.Tab("LLM Inference",elem_classes="tab-label"):
        gr.Markdown("This is the LLM Inference tab content.", elem_classes="tab-content")
        model_name=gr.Dropdown(
            get_model_list(), multiselect=True, label="model", info="Select llm."
        )
        concurrent = gr.Slider(minimum=1, maximum=100, step=1,label="concurrent", info="Choose concurrent between 1 and 100")
        gen_need_tokens = gr.Slider(minimum=1, maximum=8196, step=1024,label="Tokens", info="Choose concurrent between 1 and 8196")
        btn = gr.Button("Run")
        btn.click(llm_gpu_inference_memory_func, inputs=[model_name,concurrent, gen_need_tokens], outputs=gr.Text())
        
    with gr.Tab("LLM Train",elem_classes="tab-label"):
        model_name=gr.Dropdown(
            get_model_list(), multiselect=True, label="model", info="Select llm."
        )
        gpu_name=gr.Dropdown(
            get_gpu_list(), multiselect=True, label="gpu type", info="Select gpu type."
        )
        ##gpu_num=1,gpu_type="H100_SXM",train_tokens=9.8*pow(10,12)
        gpu_num = gr.Slider(minimum=1, maximum=10000, step=1,label="gpu_num", info="Choose gpu num between 1 and 10000")
        train_tokens = gr.Slider(minimum=1, maximum=50, step=0.1,label="Tokens", info="Choose tokens between 1 and 50 unit T")
        btn = gr.Button("Run")
        btn.click(estimate_llm_train_day, inputs=[model_name,gpu_num,gpu_name,train_tokens], outputs=gr.Text())
        
if __name__ == "__main__":
    loguru.logger.info("web ui starting")
    app.launch()
     
    