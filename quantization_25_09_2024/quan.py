import torch
from diffusers import FluxPipeline
import torch
from torch.ao.quantization import quantize_dynamic
import gc
from transformers import BitsAndBytesConfig

class Quantizer:
    def __init__(self):
        self.type = None
        self.dtype = torch.qint8
        self.size_threshold_mb = 10
    def set_model_name(self, name):
        self.type = "model"
        self.model_name = name
    def set_pipeline(self, pipeline):
        self.type = "pipe"
        self.pipeline = pipeline
        self.modules = self.get_all_modules_name(pipeline)
    def quantize(self):
        if self.type == "pipe":
            for module in self.modules:
                print("quantizing", module)
                self.quantize_pipeline(getattr(self.pipeline, module))
        elif self.type == "model":
            self.quantize_model()
    def quantize_model(self,use_4bit=True, bnb_4bit_compute_dtype="float16", 
                       bnb_4bit_quant_type= "nf4", use_nested_quant=False ):
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        
        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
                
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
        )
    def get_all_modules_name(self, pipe):
        res = []
        for k in pipe.__dict__:
            if isinstance(getattr(pipe, k), torch.nn.modules.module.Module):
                res.append(k)
        return res
    def get_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    def quantize_module(self, module):
        return quantize_dynamic(
            module,
            {torch.nn.Linear},
            dtype=self.dtype
        )
    def quantize_pipeline(self, module, parent_name=''):
        quantized = False
        for name, submodule in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            size_mb = self.get_model_size(submodule)
            
            if self.size_threshold_mb is None or size_mb > self.size_threshold_mb:
                print(f"Quantizing {full_name} ({size_mb:.2f} MB)...")
                submodule = self.quantize_module(submodule)
                setattr(module, name, submodule)
                quantized = True
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Skipping {full_name} ({size_mb:.2f} MB) - below threshold")
        return module
    def save(self, output_path):
        if self.type == "pipe":
            self.pipeline.save_pretrained(output_path)
        elif self.type == "model":
            self.model.save_pretrained(output_path)
        else:
            print("nothing to save")
class Main:
    def quantize_pipeline(pipe):
        q = Quantizer()
        q.set_pipeline(pipe)
        return q
    def quantize_model(model_name):
        q = Quantizer()
        q.set_model_name(model_name)
        return q
