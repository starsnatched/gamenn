from typing import Optional, List
from pydantic import BaseModel, Field
import yaml


class ModelConfig(BaseModel):
    pretrained: str = Field(...)
    quantization_4bit: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    max_length: int = 256
    temperature: float = 0.8
    top_k: int = 20
    use_gradient_checkpointing: bool = True


class MCTSConfig(BaseModel):
    num_sims: int = 64
    c_puct: float = 1.2
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    progressive_widening_k: float = 1.0
    progressive_widening_alpha: float = 0.5
    root_top_k: int = 64


class CurriculumConfig(BaseModel):
    enabled: bool = True
    window: int = 64
    increase_threshold: float = 0.7
    decrease_threshold: float = 0.4
    max_sims: int = 256
    min_sims: int = 16


class TrainingConfig(BaseModel):
    batch_size: int = 1
    mini_batch_size: int = 1
    ppo_epochs: int = 1
    learning_rate: float = 5e-5
    kl_target: float = 0.05
    seed: int = 42
    steps: int = 1
    rollout_steps: int = 1


class DatasetConfig(BaseModel):
    path: Optional[str] = None
    field_prompt: str = "prompt"
    max_samples: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    out_dir: str = "runs"


class Config(BaseModel):
    model: ModelConfig
    mcts: MCTSConfig
    curriculum: CurriculumConfig
    training: TrainingConfig
    dataset: DatasetConfig
    logging: LoggingConfig

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)

