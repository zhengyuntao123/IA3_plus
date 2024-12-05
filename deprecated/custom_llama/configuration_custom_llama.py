from transformers.models.llama.configuration_llama import LlamaConfig

class MyLlamaConfig(LlamaConfig):
    model_type = "custom_llama"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)