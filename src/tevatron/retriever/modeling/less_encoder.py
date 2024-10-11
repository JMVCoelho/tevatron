import torch
import logging
from .encoder import EncoderModel, EncoderOutput
from typing import Dict, Optional
from torch import nn, Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DenseModelLESS(EncoderModel):

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            reps = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, per_sample=True):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None
        
        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)

        if per_sample:    
            # negatives = p_reps[1:, :]
            # positives = p_reps[0, :].expand_as(negatives)

            # # def add_gaussian_noise(tensor, radius=0.01): # tensor: [n, embedding_dim]
            # #     norms = torch.norm(tensor, dim=1)  # Compute norms of each vector
            # #     noise = torch.randn_like(tensor)  # Generate Gaussian noise (N(0,1))
            # #     scaled_noise = noise * (radius * norms.unsqueeze(1))  # Scale noise with r% norm
            # #     tensor_with_noise = tensor + scaled_noise  # Add noise to original tensor
            # #     return tensor_with_noise
            
            # # negatives = add_gaussian_noise(negatives)

            # interleaved = torch.cat((positives.unsqueeze(1), negatives.unsqueeze(1)), dim=1)
            # interleaved = interleaved.view(-1, negatives.size(1))

            # scores = self.compute_similarity(q_reps, interleaved)
            # scores = scores.view(scores.size(1)//2, -1)

            # target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

            # loss = self.non_reduced_ce(scores / self.temperature , target)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.cross_entropy(scores/self.temperature, target)            
        else:
            print("correct")
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = F.cross_entropy(scores / self.temperature, target, reduction='mean')


        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    

    # def update_params(self, deltas): #Meta - update the buffers, not nn.Params
    #     sub_params = {}
    #     for key, delta in deltas.items():
    #         if not ('.' in key):
    #             self._buffers[key] = self._buffers[key] + delta
    #         else:
    #             attr = key.split('.')[0]
    #             if not (attr in sub_params):
    #                 sub_params[attr] = {}
    #             sub_params[attr]['.'.join(key.split('.')[1:])] = delta            
    #     for key, value in sub_params.items():
    #         self._modules[key].update_params(value)

    def update_params(self, deltas): 
        def update_params_method(self, deltas): # Define the method
            sub_params = {}
            for key, delta in deltas.items():
                if not ('.' in key):
                    self._buffers[key] = self._buffers[key] + delta
                else:
                    attr = key.split('.')[0]
                    if not (attr in sub_params):
                        sub_params[attr] = {}
                    sub_params[attr]['.'.join(key.split('.')[1:])] = delta            
            for key, value in sub_params.items():
                self._modules[key].update_params(value)

        # Dynamically set the method to self._modules[key]
        for key in self._modules:
            setattr(self._modules[key], 'update_params', update_params_method.__get__(self._modules[key]))

