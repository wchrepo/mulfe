from lmqg import TransformersQG
import json
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import os

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

model = TransformersQG("lmqg/t5-base-squad-qag")

import spacy
nlp = spacy.load("en_core_web_trf")


path = "eval_data/mulfe_test.json"
result_data = []
with open(path) as f:
    source_data = json.load(f)
    for d in tqdm(source_data):
        d = d.copy()  
        d['simplification'] = []
        for q,a in model.generate_qa(d['doc']):
            doc = nlp(q)
            if doc.ents:
                subj = doc.ents[-1].text
                left,right = f"Question: {q}\nAnswer:".rsplit(subj,1)
                if len(left)>0 and left[-1] != " ":
                    left = left + " "
                prompt = left + "{}" + right
                d['simplification'].append(
                    {
                        "input":f"Question: {q}\nAnswer:",
                        "target":f" {a}",
                        "prompt":prompt,
                        "subject":subj
                    }
                )
        result_data.append(d)

with open(path+".lmqg",'w') as f:
    json.dump(result_data,f,ensure_ascii=False,indent=2)

#post fix
# import json
# with open("eval_data/mulfe_test.json.lmqg") as f:
#     result_data = json.load(f)
#     for d in result_data:
#         for p in d['simplification']:
#             left,right = p['input'].rsplit(p['subject'],1)
#             if len(left)>0 and left[-1] != " ":
#                 left = left + " "
#             prompt = left + "{}" + right
#             p['prompt'] = prompt

# with open("eval_data/mulfe_test.json.lmqg",'w') as f:
#     json.dump(result_data,f,ensure_ascii=False,indent=2)
