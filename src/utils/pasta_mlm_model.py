from transformers import DebertaV2Model, DebertaV2Tokenizer
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import numpy as np
import random
from packaging import version
import torch

def linear_act(x):
    return x

def _mish_python(x):
    return x * torch.tanh(nn.functional.softplus(x))

if version.parse(torch.__version__) < version.parse("1.9"):
    mish = _mish_python
else:
    mish = nn.functional.mish

def gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

def _silu_python(x):
    return x * torch.sigmoid(x)


if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = nn.functional.silu

def gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = gelu_python
else:
    gelu = nn.functional.gelu

ACT2FN = {
    "relu": nn.functional.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_python": gelu_python,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}


class DebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class DebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class ModelDefine(nn.Module):
    def __init__(self, model_args):
        super(ModelDefine, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained(model_args.model_path)
        self.config = self.deberta.config
        self.cls = DebertaV2OnlyMLMHead(self.config)
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1,self.config.vocab_size), labels.view(-1))
        
        return prediction_scores, masked_lm_loss
