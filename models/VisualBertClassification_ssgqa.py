"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
from torch import nn
from models.VisualBert_ssgqa import VisualBertModel, VisualBertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel

"""
VisualBert Classification Model
"""


class VisualBertClassification(nn.Module):
    """
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    """

    def __init__(self, vocab_size, layers, n_heads, num_class=10):
        super(VisualBertClassification, self).__init__()
        VBconfig = VisualBertConfig(
            vocab_size=vocab_size,
            visual_embedding_dim=530,
            num_hidden_layers=layers,
            num_attention_heads=n_heads,
            hidden_size=1024,
        )
        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.long
        ).to(device)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1], dtype=torch.float
        ).to(device)
        # append visual features to text
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                # "visual_token_type_ids": visual_token_type_ids,
                # "visual_attention_mask": visual_attention_mask,
                "output_attentions": True,
            }
        )

        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        # inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        # inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertEncoder(**inputs)
        # classification layer
        outputs = self.classifier(outputs["pooler_output"])
        return outputs
