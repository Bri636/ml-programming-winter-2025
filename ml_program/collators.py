from dataclasses import dataclass
from typing import Any, Optional, List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

def pad(
    batch_tensors: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Custom pad that optionally *always* pads to `max_length`.
    """
    # If none is set, just find the local maximum length
    # in case some sequences are still shorter than needed.
    if max_length is None:
        max_length = max(x.size(0) for x in batch_tensors)

    # Create a (batch_size, max_length) of padding_value
    batch_size = len(batch_tensors)
    out = torch.full((batch_size, max_length), padding_value, dtype=batch_tensors[0].dtype)

    # Copy each example into out
    for i, t in enumerate(batch_tensors):
        length = min(t.size(0), max_length)
        if padding_side == "left":
            out[i, max_length - length :] = t[-length:]
        else:  # padding_side == "right"
            out[i, :length] = t[:length]

    return out


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator that optionally pads the tokenized inputs to
    a fixed `max_length`.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool = False
    max_length: Optional[int] = None       # <--- Add this!

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}

        for k in features[0].keys():

            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):

                # Convert each feature to a tensor
                to_pad = [torch.tensor(ex[k]) for ex in features]

                # 1) Possibly TRUNCATE to self.max_length
                if self.max_length is not None:
                    for i in range(len(to_pad)):
                        if to_pad[i].shape[0] > self.max_length:
                            # Truncate from the right by default
                            to_pad[i] = to_pad[i][:self.max_length]

                # 2) Decide which padding_value to use
                if self.is_encoder_decoder:
                    if (k.startswith("prompt") and k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError("Need pad_token_id for prompt inputs.")
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # 3) Pad up to either the max in `to_pad` or self.max_length
                    #    (whichever is bigger).
                    #    If you definitely want EXACTLY self.max_length, you can fix that.
                    padded_batch[k] = pad_sequence(
                        to_pad,
                        batch_first=True,
                        padding_value=padding_value
                    )

                    # If you want exactly self.max_length in all cases,
                    # even if the max length is smaller, see tip below.

                else:
                    # Non-encoder-decoder
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError("Need pad_token_id for input_ids")
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Decide left vs. right padding
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    padded_batch[k] = pad(
                        to_pad,
                        padding_value=padding_value,
                        padding_side=padding_side,
                        max_length=self.max_length,  # <--- We'll modify our pad() to accept max_length
                    )

            elif k.endswith("_logps"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch
