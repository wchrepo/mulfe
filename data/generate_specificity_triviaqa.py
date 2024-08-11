

import os

import transformers
import datasets
import torch
import json
import math
from tqdm.auto import tqdm
from functools import partial


def reduce_aliases(aliases):
    aliases = sorted(aliases)
    results = []
    prefix = None
    for alias in aliases:
        if prefix is None or not alias.startswith(prefix):
            results.append(alias)
            prefix = alias
    return results



def get_prompt_with_prefix(query,prefix):
    return prefix+example_prompt_template.format(query=query)

def tokenize_prefix_and_target(tokenizer,prefixes,targets,max_length=None):
    if prefixes:
        prefix_lengths = [len(s) for s in tokenizer(prefixes,add_special_tokens=False).input_ids]
        tokenized = tokenizer([p+t for p,t in zip(prefixes,targets)],
            return_tensors='pt',
            padding=True,truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        labels = tokenized.input_ids.clone()
        labels[labels==tokenizer.pad_token_id]=-100
        if tokenizer.padding_side == 'left':
            starts = tokenized.attention_mask.size(-1) - tokenized.attention_mask.sum(-1)
        else:
            starts = [0]*len(prefix_lengths)
        for i in range(len(prefix_lengths)):
            start = starts[i]
            labels[i,start:start+prefix_lengths[i]]=-100
        tokenized['labels']=labels
    else:
        tokenized = tokenizer(targets,
            return_tensors='pt',
            padding=True,truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        labels = tokenized.input_ids.clone()
        labels[labels==tokenizer.pad_token_id]=-100
        tokenized['labels']=labels
    return tokenized

def collate_fn(batch):
    #batch = [(qid,query,answer)]
    #return（qids, tensor batch）
    qids,queries,answers = zip(*[(e['id'],e['query'],e['answer']) for e in batch])
    inputs = []
    for q in queries:
        inputs.extend([get_prompt_with_prefix(q,p) for p in zero_shot_prompts])
    targets = []
    for a in answers:
        targets.extend([" "+a.strip() for p in zero_shot_prompts])
    return (qids,tokenize_prefix_and_target(tokenizer,inputs,targets))

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
    #mean on different prompts
    correct = correct.reshape(-1,len(zero_shot_prompts)).float().mean(-1)
    nll = nll.reshape(-1,len(zero_shot_prompts)).mean(-1)
    label_lengths = label_lengths.reshape(-1,len(zero_shot_prompts))[:,0]
    results = list(zip(qids,correct.tolist(),nll.tolist(),label_lengths.tolist()))
    return results

# def batchify(data,batch_size):
#     for i in range(0,len(data),batch_size):
#         yield collate_fn(data[i:i+batch_size])

generate_kwargs = dict(
    max_new_tokens=30
)
def generate_pipeline(input_text,**kwargs):
    batch = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(device=model.device)
    results = model.generate(**batch,**generate_kwargs,**kwargs)
    results = tokenizer.batch_decode(results[:,batch.input_ids.size(1):])
    # results = run_batch_greedymatch(model,batch)
    return results

def fast_test_example(i):
    return generate_pipeline(get_prompt(simplified_data[i]['query'])),simplified_data[i]['answer']

# model_name = "gpt2-xl"
#model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "EleutherAI/gpt-neo-2.7B"
model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "NousResearch/Llama-2-7b-chat-hf"
# device = 'cuda:1'
# device_map = None
device_map = 'auto'
batch_size= 16
consider_alias = True 

if "Llama-2" in model_name and "chat" in model_name:
    BOS = '<s>'
    EOS = '</s>'
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    zero_shot_prompt = f"{BOS}{B_INST} {B_SYS}Directly answer the question without adding any other content.{E_SYS}"
    zero_shot_prompts = [
        f"{BOS}{B_INST} {B_SYS}Directly answer the question without adding any other content.{E_SYS}",
        f"{BOS}{B_INST} {B_SYS}Answer the question without adding any other content.{E_SYS}",
        f"{BOS}{B_INST} {B_SYS}Provide an answer exclusively to the question asked.{E_SYS}"
    ]
    # zero_shot_prompt = f"{BOS}{B_INST} {B_SYS}Directly answer the question.{E_SYS}"
    example_prompt_template = f"{{query}} {E_INST} Answer:"
else:
    zero_shot_prompt = "Directly answer the question.\n\n"
    zero_shot_prompts = [
        "Directly answer the question.\n\n",
        "Answer the question without adding any other content.\n\n",
        "Provide an answer exclusively to the question asked.\n\n"
    ]
    example_prompt_template = "Question: {query}\nAnswer:"

get_prompt = partial(get_prompt_with_prefix,prefix=zero_shot_prompt)

data = datasets.load_dataset('trivia_qa','rc.wikipedia.nocontext',split='validation')
simplified_data = []

if consider_alias:
    for d in data:
        for alias in reduce_aliases([d['answer']['value']]+d['answer']['aliases']):
            simplified_data.append({"query":d['question'],"answer":alias,'id':d['question_id']})
else:
    for d in data:
        simplified_data.append({"query":d['question'],"answer":d['answer']['value'],'id':d['question_id']})

model = transformers.AutoModelForCausalLM.from_pretrained(model_name,device_map=device_map)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
# model.to(device=device)

results = []
for batch in tqdm(torch.utils.data.DataLoader(simplified_data,batch_size=batch_size,collate_fn=collate_fn)):
    with torch.no_grad():
        torch.cuda.empty_cache()
        results.extend(run_batch_greedymatch(model,batch))
filtered_data = []
qid_set = set()
for i,r in enumerate(results):
    if r[1]==1 and r[0] not in qid_set:
        filtered_data.append(simplified_data[i])
        filtered_data[-1]['metrics'] = {
            "nll":r[2],
            "ppl":math.exp(r[2]/r[3])
        }
        qid_set.add(r[0])


with open("eval_data/tq_specifity/"+model_name.replace("/","_")+".json",'w') as f:
    json.dump(filtered_data,f,ensure_ascii=False,indent=2)


# 筛选400个
import random
import os,json
for name in os.listdir("eval_data/tq_specifity"):
    with open(f"eval_data/tq_specifity/{name}") as f:
        data = json.load(f)
        # # 不随机了，而是采用ppl排序从小往大排序
        # data.sort(key=lambda x:x['metrics']['ppl'])
        #随机
        random.seed(0)
        random.shuffle(data)
    with open(f"eval_data/tq_specifity_random400/{name}",'w') as f:
        json.dump(data[:400],f,ensure_ascii=False,indent=2)




