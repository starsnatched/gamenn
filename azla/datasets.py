from typing import Dict, Iterable, List
import json


from typing import Optional


def load_jsonl_prompts(path: str, field_prompt: str, max_samples: Optional[int] = None) -> List[str]:
    items: List[str] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if field_prompt in obj:
                items.append(obj[field_prompt])
            if max_samples and len(items) >= max_samples:
                break
    return items
