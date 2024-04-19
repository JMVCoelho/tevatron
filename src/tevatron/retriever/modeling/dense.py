import torch
import logging
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def encode_query(self, qry, is_query=True):
        # if self.repetition:
        #     self.encoder.eval()
        #     with torch.no_grad():
        #         query_hidden_states = self.encoder(**qry, return_dict=True)
        #         query_hidden_states = query_hidden_states.last_hidden_state
        #         sequence_lengths = qry['attention_mask'].sum(dim=1) - 1 
        #         batch_size = query_hidden_states.shape[0]
        #         summary_ids = query_hidden_states[torch.arange(batch_size, device=query_hidden_states.device), sequence_lengths]
        #         del query_hidden_states

        #     self.encoder.train()
        #     query_hidden_states = self.encoder(**qry, summary_ids=summary_ids, return_dict=True)
        #     query_hidden_states = query_hidden_states.last_hidden_state
        #     return self._pooling(query_hidden_states, qry['attention_mask'], is_query=is_query)
        # else:
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'], is_query=is_query)
    
    def encode_passage(self, psg):
        return self.encode_query(psg, is_query=False)
        
    def _pooling(self, last_hidden_state, attention_mask, is_query):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            reps = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1)

            if not self.repetition:
                sequence_lengths -= 1

            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        elif self.pooling in ['landmark']:
            # print(last_hidden_state.shape)
            # print(attention_mask.shape)
            # print(attention_mask.sum(dim=1).shape)
            # exit()
            if is_query:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            else:
                batch_size = last_hidden_state.shape[0]
                reps = []
                for i in range(batch_size):
                    num_tokens = attention_mask[i].sum()
                    landmarks = [(num_tokens - ((num_tokens//4) * j)) - 1 for j in range(4)]
                    reps_per_input = []
                    for landmark_pos in landmarks:
                        reps_per_input.append(last_hidden_state[i, landmark_pos])
                    reps.append(torch.stack(reps_per_input))

                reps = torch.stack(reps)
                doc_embeddings = torch.mean(reps, dim=1)
                doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
                ldmark_reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
                
                return doc_embeddings, ldmark_reps

                
                
                
                #landmark_reduced_embeddings = self.ld_head(reps)
                #reps = landmark_reduced_embeddings.view(landmark_reduced_embeddings.shape[0], 
                #                                                        landmark_reduced_embeddings.shape[1]*landmark_reduced_embeddings.shape[2])
                
                
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
