import torch

class MaskedDiffusionCollator:
    def __init__(self, tokenizer, mask_token_id=None, pad_token_id=None, min_mask_rate=0.02, max_mask_rate=0.98):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id if mask_token_id is not None else tokenizer.mask_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        if self.mask_token_id is None:
            self.mask_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
        self.min_mask_rate = min_mask_rate
        self.max_mask_rate = max_mask_rate

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = input_ids.clone()
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        # Continuous noise schedule (Rectified Flow / MDLM)
        t = torch.rand(bsz, 1, device=device)
        mask_ratios = self.min_mask_rate + (self.max_mask_rate - self.min_mask_rate) * t
        rand_matrix = torch.rand(bsz, seq_len, device=device)
        mask_indices = rand_matrix < mask_ratios

        # Never mask padding or special tokens
        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()],
            dtype=torch.bool, device=device
        )
        mask_indices = mask_indices & ~special_tokens_mask

        # -100 on uncorrupted = PyTorch ignores them in loss
        labels[~mask_indices] = -100
        corrupted_input_ids = input_ids.clone()
        corrupted_input_ids[mask_indices] = self.mask_token_id

        # 4D attention mask for bidirectional patches
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.bfloat16).min

        return {
            "input_ids": corrupted_input_ids,
            "attention_mask": extended_attention_mask,
            "labels": labels
        }
