import requests
import json
from unidecode import unidecode
from tqdm.auto import tqdm

path = "eval_data/mulfe_test.json"
with open(path) as f:
    source_data = json.load(f)

url = "http://localhost:5000/build"
result_data = []
for d in tqdm(source_data):
    d = d.copy()  
    r = requests.post(url,                                                     
                    data=json.dumps({'text': unidecode(d['doc'])}),
                    headers={'Content-type': 'application/json'})  
    response = json.loads(r.text)
    d['simplification'] = []
    if response:
        processed_triples = response['proc'] # Processed semantic triple after term selection & alignment
        for triple in processed_triples:
            d['simplification'].append(
                {
                    'prompt':"{} "+triple['rel'],
                    "input":triple['sub'] + " " + triple['rel'],
                    "target":" "+triple['obj'],
                    "subject":triple['sub'],
                }
            )
    result_data.append(d)

with open(path+".doc_openie",'w') as f:
    json.dump(result_data,f,ensure_ascii=False,indent=2)
# # Compare the original triple with the processed triple
# for ori, proc in zip(preprocessed_triples,processed_triples):
#     print(f"Original : {ori['sub']} --- {ori['rel']} --> {ori['obj']}")
#     print(f"Processed: {proc['sub']} --- {proc['rel']} --> {proc['obj']}") 
#     print("")