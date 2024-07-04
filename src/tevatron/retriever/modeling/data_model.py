import torch
import torch.nn as nn
from .rope_t5 import T5ModelRoPEFusion, T5ModelRoPE
from transformers import BertModel
from torch.nn import MSELoss, CrossEntropyLoss



from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

class T5FusionRegression(nn.Module):
    def __init__(self, model_path, n_fusion):
        super().__init__()
        self.t5fid = T5ModelRoPEFusion.from_pretrained(model_path, n_fusion)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(self.t5fid.config.hidden_size, 1)
        self.loss_fct = MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long).to(input_ids.device)
        outputs = self.t5fid(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        hidden = outputs.last_hidden_state
        reps = self.dropout(hidden[:, 0, :]) # bert for seq classification does this.
        logits = self.regression(reps)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.flatten(), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.decoder_hidden_states,
            attentions=outputs.cross_attentions,
        )


class BERTBinaryClassification(nn.Module):
    def __init__(self, model_path, n_fusion):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(self.bert.config.hidden_size, 2)
        self.loss_fct = CrossEntropyLoss()
        self.n_fusion = n_fusion
    
    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(last_hidden_states[:, :, 0])
        
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state #[bs*fusion, seq_len, hidden_dim]

        ## AVG pool each independent q/d, then avg pool everything
        pooled = self.average_pool(hidden, attention_mask)
        per_example = pooled.reshape(pooled.size(0)//self.n_fusion, -1, pooled.size(-1))
        reps = self.average_pool(per_example)

        ## Just avgpool everything at once
        # pooled = hidden.reshape(hidden.size(0)//self.n_fusion, -1, hidden.size(-1))
        # attention_mask = attention_mask.reshape(-1, pooled.size(1))
        # reps = self.average_pool(pooled, attention_mask)

        reps = self.dropout(reps)
        logits = self.regression(reps)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class BERTRegression(nn.Module):
    def __init__(self, model_path, n_fusion):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fct = MSELoss()
        self.n_fusion = n_fusion
    
    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(last_hidden_states[:, :, 0])
        
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state #[bs*fusion, seq_len, hidden_dim]

        ## AVG pool each independent q/d, then avg pool everything
        # pooled = self.average_pool(hidden, attention_mask)
        # per_example = pooled.reshape(pooled.size(0)//self.n_fusion, -1, pooled.size(-1))
        # reps = self.average_pool(per_example)

        ## Just avgpool everything at once
        pooled = hidden.reshape(hidden.size(0)//self.n_fusion, -1, hidden.size(-1))
        attention_mask = attention_mask.reshape(-1, pooled.size(1))
        reps = self.average_pool(pooled, attention_mask)

        reps = self.dropout(reps)
        logits = self.regression(reps)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.flatten(), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class T5Regression(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.t5fid = T5ModelRoPE.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(self.t5fid.config.hidden_size, 1)
        self.loss_fct = MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long).to(input_ids.device)
        outputs = self.t5fid(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        hidden = outputs.last_hidden_state
        reps = self.dropout(hidden[:, 0, :]) # bert for seq classification does this.
        logits = self.regression(reps)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.flatten(), labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.decoder_hidden_states,
            attentions=outputs.cross_attentions,
        )