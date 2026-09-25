from __future__ import annotations

import numpy as np

BACKBONE = "facebook/esm2_t33_650M_UR50D"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
LORA_TARGETS = ("query", "key", "value")
LR = 1e-4
BATCH_TOKENS = 16384
TRAIN_BATCH_TOKENS = 3072

def _mean_pool(hidden, attention_mask):
    import torch

    mask = attention_mask.clone()
    mask[:, 0] = 0
    last = attention_mask.sum(1) - 1
    mask[torch.arange(mask.size(0), device=mask.device), last] = 0
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1)

class Esm2Encoder:
    def __init__(self, device: str = "cuda:0"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(BACKBONE)
        self.model = AutoModel.from_pretrained(BACKBONE, dtype=torch.float32).to(device).eval()
        self.fp0 = fingerprint(self.model)

    def embed_mean(self, sequences: list[str]) -> np.ndarray:
        return mean_pool(self.model, self.tok, sequences, self.device)

    def embed_frozen(self, sequences: list[str]) -> np.ndarray:
        return embed(self.model, self.tok, sequences, self.device)

    def adapt_and_embed(self, train_seqs, train_targets, embed_seqs, *,
                        epochs: int, seed: int) -> np.ndarray:
        import torch

        adapted = finetune(
            self.model, self.tok, list(train_seqs), train_targets, self.device,
            seed=seed, epochs=epochs,
        )
        try:
            out = embed(adapted, self.tok, list(embed_seqs), self.device)
        finally:
            adapted.unload()
            del adapted
            torch.cuda.empty_cache()
        if abs(fingerprint(self.model) - self.fp0) > 1e-3:
            raise RuntimeError("base weights moved: LoRA merged, labels would leak")
        return out

def fingerprint(model) -> float:
    import torch

    with torch.no_grad():
        values = [
            p.detach().float().sum().item()
            for name, p in model.named_parameters()
            if "lora" not in name
        ]
    return float(np.sum(values))

def embed(model, tok, seqs, device, batch_tokens: int = BATCH_TOKENS, *,
          normalize: bool = True) -> np.ndarray:
    import torch

    order = np.argsort([len(s) for s in seqs])
    ordered = [seqs[j] for j in order]
    chunks, i = [], 0
    while i < len(ordered):
        length = len(ordered[i]) + 2
        batch = max(1, min(batch_tokens // length, len(ordered) - i))
        enc = tok(ordered[i:i + batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        chunks.append(_mean_pool(hidden, enc["attention_mask"]).float().cpu().numpy())
        i += batch
    stacked = np.concatenate(chunks, axis=0)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    embedded = stacked[inverse].astype(np.float64)
    if not normalize:
        return embedded
    return embedded / np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)

def mean_pool(model, tok, seqs, device, batch_tokens: int = BATCH_TOKENS) -> np.ndarray:

    return embed(model, tok, seqs, device, batch_tokens, normalize=False)

def finetune(model, tok, seqs, targets, device, seed: int, epochs: int):
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(int(seed))
    adapted = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=list(LORA_TARGETS),
    ))
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad] + list(head.parameters()),
        lr=LR,
    )
    target = torch.tensor(np.asarray(targets, dtype=np.float32), device=device)
    target = (target - target.mean()) / (target.std() + 1e-8)
    adapted.train()
    seqs = list(seqs)
    n = len(seqs)
    order = sorted(range(n), key=lambda i: -len(seqs[i]))
    chunks, cur, cur_len = [], [], 0
    for i in order:
        length = len(seqs[i]) + 2
        if cur and (len(cur) + 1) * max(cur_len, length) > TRAIN_BATCH_TOKENS:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len = max(cur_len, length)
    if cur:
        chunks.append(cur)
    for _ in range(int(epochs)):
        opt.zero_grad()
        for chunk in chunks:
            enc = tok([seqs[i] for i in chunk], return_tensors="pt", padding=True).to(device)
            hidden = adapted(**enc).last_hidden_state
            pred = head(_mean_pool(hidden, enc["attention_mask"])).squeeze(-1)
            loss = torch.nn.functional.mse_loss(
                pred, target[torch.tensor(chunk, device=device)]
            ) * (len(chunk) / n)
            loss.backward()
            del hidden, enc, pred, loss
        opt.step()
    adapted.eval()
    return adapted
