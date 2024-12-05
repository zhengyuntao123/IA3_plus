from transformers import LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel
import torch.nn as nn
import torch
from typing import Type, Optional, Tuple
from .configuration_custom_llama_residual import MyLlamaConfig

def apply_to_all_named_modules(module: nn.Module, fn, parent_name: str = ""):
    '''Recursively applies a function to all named modules in a PyTorch module.'''
    # Recurse through children with their instance names
    for name, child in module.named_children():
        # Construct the full name path for the current module
        full_name = parent_name + ("." if parent_name else "") + name
        # Apply the function to the current module
        fn(full_name, module, name, child)
        # Recurse into the child module
        apply_to_all_named_modules(child, fn, full_name)

def print_model_layers(model: nn.Module):
    '''Recursively prints the variable names of all layers in a PyTorch model and their type.'''
    apply_to_all_named_modules(
        model,
        lambda full_name, module, name, child: print(f"{full_name}: {child.__class__.__name__}")
    )

def replace_module_by_class_and_name(module: Type[nn.Module],
                                     target_class: str,
                                     target_name: str,
                                     replacement_class: Type[nn.Module],
                                     other_init_args: Tuple = ()):
    '''
    替换类名为target_class, 实例名target_name为的模块
    '''

    # Lambda function used to replace the target module with the replacement module
    def replace_module_by_class_and_name_fn(full_name, module, name, child):
        # print(f"{full_name}: {child.__class__.__name__}")
        # If the current module is of the target class, replace it
        if name == target_name and child.__class__.__name__ == target_class:
            print("Replacing: ", target_class, replacement_class)
            # 用原本的attention层初始化
            setattr(module, name, replacement_class(child, *other_init_args))

    # Recursively apply the replacement function to all named modules
    apply_to_all_named_modules(
        module,
        replace_module_by_class_and_name_fn,
    )

class MyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(MyRMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim)) # 把scale初始化为0

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed

class MyLinear(nn.Linear):
    # 之所以要继承nn.Linear是因为tuners\ia3\model.py中的_create_new_module定死了只能接受target_base_layer为torch.nn.Linear
    # 设置为(1,1)和out_features是为了(几乎)不增加参数量，因为虽然我们继承了nn.Linear，但根本不会去用其中的参数
    def __init__(self, old_linear: nn.Linear, out_features):
        super().__init__(1,1,bias=False)
        self.linear = old_linear
        self.rms_norm=MyRMSNorm(out_features, eps=1e-6)

    def forward(self, x):
        x = self.linear(x)
        return self.rms_norm(x)+x

class CustomLlamaModel(LlamaModel):
    config_class = MyLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        # Replace 'q_proj' and 'k_proj' layers with 'MyLinear'
        replace_module_by_class_and_name(self.layers, 'Linear', 'q_proj', MyLinear, (2048,))
        replace_module_by_class_and_name(self.layers, 'Linear', 'k_proj', MyLinear, (512,))
        # Initialize weights and apply final processing
        self.post_init()

    def apply_custom_modifications(self):
        def replace_module_by_class_and_name(module: nn.Module,
                                             target_class: str,
                                             target_name: str,
                                             replacement_class: Type[nn.Module],
                                             other_init_args: Tuple = ()):
            def replace_module_by_class_and_name_fn(full_name, module, name, child):
                if name == target_name and child.__class__.__name__ == target_class:
                    setattr(module, name, replacement_class(child, *other_init_args))
            apply_to_all_named_modules(module, replace_module_by_class_and_name_fn)

class CustomLlamaForCausalLM(LlamaForCausalLM):
    config_class = MyLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        self.post_init()

    def save_checkpoint(self, dir):
        # to bypass the code line 2291 in transformers.trainer
        pass





