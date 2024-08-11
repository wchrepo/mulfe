
import os

import transformers
import datasets
import torch
import json
import math
from tqdm.auto import tqdm
from functools import partial
from utils import tokenize_prefix_and_target




def get_prompt_with_prefix(query,context="", answer_start="",prefix='',example_prompt_template="{query}"):
    return prefix+example_prompt_template.format(query=query,context=context,answer_start=answer_start)

def get_cloze_hint(q):
    hint = ""
    if "___" in q and not "?" in q:
        hint = q.split("___")[0]
        if len(hint)>0:
            hint = " "+hint.strip()
    return hint

def run_batch_greedymatch(model,batch):
    NULL_TOKEN = 0
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
    pred_ids = logits.argmax(-1).masked_fill(~label_mask, NULL_TOKEN)
    targ_ids = labels.masked_fill(~label_mask,NULL_TOKEN)
    correct = (pred_ids == targ_ids).all(dim=-1)
    results = list(zip(qids,correct.tolist(),nll.tolist(),label_lengths.tolist()))
    return results

def run_batch_multichoice(model,batch):
    qids,batch = batch
    batch.to(device=model.device)
    outputs = model(**batch)
    logits = outputs.logits
    logits = logits[:,:-1]
    labels = batch['labels']
    labels = labels[:,1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),reduction="none").view(logits.size(0),-1)
    ll = -loss.sum(dim=-1)
    results = list(zip(qids,ll.tolist()))
    return results
# generate_kwargs = dict(
#     max_new_tokens=30
# )
# def generate_pipeline(input_text,**kwargs):
#     batch = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(device=model.device)
#     results = model.generate(**batch,**generate_kwargs,**kwargs)
#     results = tokenizer.batch_decode(results[:,batch.input_ids.size(1):])
#     # results = run_batch_greedymatch(model,batch)
#     return results

# def fast_test_example(i):
#     return generate_pipeline(get_prompt(simplified_data[i]['query'])),simplified_data[i]['answer']

class QAPipeline:
    def __init__(self,model,tokenizer,batch_size=32,eval_type="greedymatch",special_template="default",with_ctx=False,add_cloze_hint=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eval_type = eval_type
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size
        self.add_cloze_hint = add_cloze_hint
        if special_template == 'llama2-chat':
            BOS = '<s>'
            EOS = '</s>'
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            if special_template == 'llama2-chat' and with_ctx:
                prefix_prompt = f"{BOS}{B_INST} {B_SYS}Directly answer the question without adding any other content.{E_SYS}"
                example_prompt_template = f"{{context}}\n\nQuestion: {{query}} {E_INST} Answer:{{answer_start}}"
            elif special_template == 'llama2-chat' and not with_ctx:
                prefix_prompt = f"{BOS}{B_INST} {B_SYS}Directly answer the question without adding any other content.{E_SYS}"
                example_prompt_template = f"{{query}} {E_INST} Answer:{{answer_start}}"
        elif special_template == 'blank':
            #for entity inferences dataset
            prefix_prompt = ""
            if with_ctx:
                example_prompt_template = "{context} {query}"
            else:
                example_prompt_template = "{query}"
        else:
            if with_ctx:
                prefix_prompt = "Directly answer the question.\n\n"
                example_prompt_template = "{context}\n\nQuestion: {query}\nAnswer:{answer_start}"
            else:
                prefix_prompt = "Directly answer the question.\n\n"
                example_prompt_template = "Question: {query}\nAnswer:{answer_start}"
        self.get_prompt = partial(get_prompt_with_prefix,prefix=prefix_prompt,
            example_prompt_template=example_prompt_template)
        
    def evaluate(self,data,eval_type=None,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if eval_type is None:
            eval_type = self.eval_type
        #data: list of probes
        if eval_type == 'greedymatch':
            results = []
            for i in range(0,len(data),batch_size):
                batch = self.collate_fn(data[i:i+batch_size])
                results.extend(run_batch_greedymatch(self.model,batch))
            qid_metrics = []
            for r in results:
                qid = r[0]
                metrics = {
                    "acc":int(r[1]),
                    "nll":r[2],
                    "tokens":r[3],
                    "ppl":math.exp(r[2]/r[3])
                }
                qid_metrics.append({"id":qid,"metrics":metrics})
            return qid_metrics
        elif eval_type == 'multichoice':
            flatten_data = []
            for d in data:
                for i,choice in enumerate(d['choices']):
                    flatten_data.append({
                        "id":(d['id'],i),
                        'query':d['query'],
                        'answer':choice
                    })
                    if "context" in d:
                        flatten_data[-1]['context'] = d['context']
            results = []
            for i in range(0,len(flatten_data),batch_size):
                batch = self.collate_fn(flatten_data[i:i+batch_size])
                results.extend(run_batch_multichoice(self.model,batch))
            qid_metrics = []
            start = 0
            for d in data:
                end = start + len(d['choices'])
                scores = [math.exp(r[1]) for r in results[start:end]]
                pred = max(range(len(scores)),key=scores.__getitem__)
                correct = pred == d['answer_id']
                metrics = {
                    "acc":int(correct),
                    "scores":scores
                }
                qid_metrics.append({"id":d['id'],"metrics":metrics})
                start = end
            return qid_metrics
    def collate_fn(self,batch):
        #batch = [(qid,query,answer)]
        #return（qids, tensor batch）
        qids,queries,answers,contexts = zip(*[(e['id'],e['query'],e['answer'],e.get('context','')) for e in batch])
        if self.add_cloze_hint:
            queries = [self.get_prompt(q,c,get_cloze_hint(q)) for q,c in zip(queries,contexts)]
        else:
            queries = [self.get_prompt(q,c) for q,c in zip(queries,contexts)]
        answers = [" "+answer.strip() for answer in answers]
        return (qids,tokenize_prefix_and_target(self.tokenizer,queries,answers))

    def debug(self,d):
        return self.collate_fn([d]),self.generate_pipeline(self.get_prompt(d['query'])),d['answer']
    
    def generate_pipeline(self,input_text,**kwargs):
        generate_kwargs = dict(
            max_new_tokens=30
        )
        batch = self.tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(device=self.model.device)
        results = self.model.generate(**batch,**generate_kwargs,**kwargs)
        results = self.tokenizer.batch_decode(results[:,batch.input_ids.size(1):])
        # results = run_batch_greedymatch(model,batch)
        return results


            



