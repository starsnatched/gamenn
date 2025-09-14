from typing import Dict, List, Tuple, Optional
import contextlib

from .utils import detect_device, to_device
from .auth import resolve_hf_token, format_auth_error


class ModelManager:
    def __init__(
        self,
        model_name: str,
        quantization_4bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_gradient_checkpointing: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.quant4 = quantization_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.device = device or detect_device()
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import BitsAndBytesConfig

        quant_args = {}
        if self.quant4:
            quant_args = {
                # "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                "device_map": "auto",
            }
        token = resolve_hf_token()
        tok_kwargs = {"use_fast": True}
        if token:
            tok_kwargs["token"] = token
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
        except Exception as e:
            raise RuntimeError(format_auth_error(self.model_name)) from e
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        mdl_kwargs = dict(quant_args)
        if token:
            mdl_kwargs["token"] = token
        try:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name, **mdl_kwargs)
        except Exception as e:
            raise RuntimeError(format_auth_error(self.model_name)) from e
        if self.use_gradient_checkpointing:
            try:
                self.model.pretrained_model.gradient_checkpointing_enable()
            except Exception:
                pass
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
        )
        self.model.pretrained_model = get_peft_model(self.model.pretrained_model, lora_cfg)
        self.model.to(self.device)
        self.model.eval()

    @contextlib.contextmanager
    def enable_dropout(self):
        import torch

        prev = self.model.training
        self.model.train()
        try:
            yield
        finally:
            self.model.train(prev)

    def tokenize(self, texts: List[str], max_length: int) -> Dict[str, "torch.Tensor"]:
        import torch

        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: to_device(v, self.device) for k, v in toks.items()}
        return toks

    def forward(self, input_ids, attention_mask, use_dropout: bool = False):
        import torch

        ctx = self.enable_dropout() if use_dropout else contextlib.nullcontext()
        with ctx:
            out = self.model.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=True)
        return out

    def logits_for_next(self, input_ids, attention_mask, use_dropout: bool = False):
        out = self.forward(input_ids, attention_mask, use_dropout=use_dropout)
        logits = out.logits[:, -1, :]
        return logits

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
