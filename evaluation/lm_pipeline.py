import torch
import math
from utils import tokenize_prefix_and_target

def run_batch_lm(model,batch):
    qids,batch = batch
    batch.to(device=model.device)
    outputs = model(**batch)
    logits = outputs.logits
    logits = logits[:,:-1]
    labels = batch['labels']
    labels = labels[:,1:]
    label_mask = (labels!=-100)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),reduction="none").view(logits.size(0),-1)
    nll = loss.sum(dim=-1)
    label_lengths = label_mask.sum(dim=-1)
    results = list(zip(qids,nll.tolist(),label_lengths.tolist()))
    return results
class LMPipeline:
    def __init__(self,model,tokenizer,batch_size=32,with_ctx=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size
        self.with_ctx = with_ctx
    

    def evaluate(self,data,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        results = []
        for i in range(0,len(data),batch_size):
            batch = self.collate_fn(data[i:i+batch_size])
            results.extend(run_batch_lm(self.model,batch))
            id_metrics = []
            for r in results:
                docid = r[0]
                metrics = {
                    "nll":r[1],
                    "tokens":r[2],
                    "ppl":math.exp(r[1]/r[2])
                }
                id_metrics.append({"id":docid,"metrics":metrics})
            return id_metrics
        
       
    def collate_fn(self,batch):
        #batch = [(qid,query,answer)]
        #返回（qids, tensor batch）
        qids,texts,contexts = zip(*[(e['id'],e['doc'],(e['context']+" " if "context" in e else "")) for e in batch])
        return (qids,tokenize_prefix_and_target(self.tokenizer,prefixes=contexts,targets=texts))