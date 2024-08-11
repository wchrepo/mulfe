
import json
import transformers
import mlflow
import os
from utils import flatten_dict
from omegaconf import OmegaConf
from qa_pipeline import QAPipeline
from lm_pipeline import LMPipeline
from summary import summarize
from functools import partial
from edit_wrappers import *
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
import random
import argparse
import datetime

batch_edit = 1 # must be 1
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--batch_edit', type=int, default=1, help='Description of batch_edit')
parser.add_argument('--experiment_name', type=str, default="FullEval-mulfe",help='Description of experiment_name')
parser.add_argument('--model_name', type=str, default="gpt2-xl", help='Description of config')
parser.add_argument('--update_config', type=str,nargs='*', default=[], help='update some config. e.g. editor.opt_kwargs.lr=1e-4')

args = parser.parse_args()
experiment_name = args.experiment_name
model_name = args.model_name
update_config = args.update_config
config = OmegaConf.merge(OmegaConf.load("config/config.yaml"),
                         OmegaConf.from_dotlist(update_config))
config.batch_edit = batch_edit
config.model_name = model_name
config.editor = {"type":"EditInContext"}

run_name = f"{config.model_name.replace('/','_')}_{config.editor.type}_e{batch_edit}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
with open("../dataeval_data/mulfe_test.json") as f:
    data = json.load(f)
with open(f"../dataeval_data/{config.specifity_data}/{config.model_name.replace('/','_')}.json") as f:
    specifity_data = json.load(f)

eval_chunks = []
random.seed(0)
random.shuffle(data) 
for start in range(0,len(data),batch_edit):
    chunk = {
        "edit":data[start:start+batch_edit],
        "original":data[start:start+batch_edit],
        "probes": sum([d['probes'] for d in data[start:start+batch_edit]],[]),
        "specifity_probes": specifity_data
    }
    chunk = deepcopy(chunk)
    assert batch_edit == 1
    for d in chunk['original']:
        d['context'] = data[start]['doc']
    for d in chunk['probes']:
        d['context'] = data[start]['doc']
    for d in chunk['specifity_probes']:
        d['context'] = data[start]['doc']
    eval_chunks.append(chunk)
special_template = "default"
if "Llama-2" in config.model_name and "chat" in config.model_name:
    special_template = 'llama2-chat'
group_evaluator = {
    "original":partial(LMPipeline,with_ctx=True),
    "probes":partial(QAPipeline,eval_type="greedymatch",special_template=special_template,with_ctx=True,add_cloze_hint=config.add_cloze_hint),
    "specifity_probes":partial(QAPipeline,eval_type="greedymatch",special_template=special_template,with_ctx=True),
}

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,device_map='auto')
tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
editor = NoEdit(model,tokenizer,config)

mlflow.set_experiment(experiment_name)
mlflow.start_run(run_name=run_name)
mlflow.log_params(flatten_dict(config))

results = {k:[] for k in group_evaluator}
chunk_outputs = []
for chunk in tqdm(eval_chunks,dynamic_ncols=True):
    with editor.autorestore():
        editor.edit(chunk['edit'])
        with torch.no_grad():
            editor.model.eval()
            chunk_results = {}
            for key,evaluator_class in group_evaluator.items():
                evaluator = evaluator_class(editor.model,editor.tokenizer)
                chunk_results[key] = evaluator.evaluate(chunk[key])
                results[key].extend(chunk_results[key])
            chunk_outputs.append({"edit":chunk['edit'],"outputs":chunk_results})

#subdivide level
level_mapping = {}
for d in data:
    for p in d['probes']:
        level_mapping[p['id']] = p['level']
for r in results['probes']:
    results.setdefault(f"level_{level_mapping[r['id']]}",[]).append(r)
summaries = {k:summarize(v) for k,v in results.items() }
print(summaries)
mlflow.log_metrics(flatten_dict(summaries))
chunk_outputs_json = json.dumps(chunk_outputs,ensure_ascii=False,indent=2)
mlflow.log_text(chunk_outputs_json,"chunk_outputs.json")

os.makedirs(f"outputs/{experiment_name}/{run_name}/",exist_ok=True)
with open(f"outputs/{experiment_name}/{run_name}/chunk_outputs.json","w") as f:
    f.write(chunk_outputs_json)
with open(f"outputs/{experiment_name}/{run_name}/metrics.json","w") as f:
    json.dump(summaries,f,ensure_ascii=False)
with open(f"outputs/{experiment_name}/{run_name}/config.yaml","w") as f:
    f.write(OmegaConf.to_yaml(config))
    




        



