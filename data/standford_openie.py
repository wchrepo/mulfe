import requests
import json
from tqdm.auto import tqdm
# from openie import StanfordOpenIE
import stanza
from stanza.server import CoreNLPClient

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 1 / 3,
    'openie.resolve_coref':True,
    # 'openie.format':"ollie"
}

# stanza.install_corenlp()





path = "eval_data/mulfe_test.json"
with open(path) as f:
    source_data = json.load(f)


with CoreNLPClient(properties=properties,endpoint="http://localhost:19000",annotators=["coref","openie"], be_quiet=False) as client:
    result_data = []
    for d in tqdm(source_data):
        d = d.copy()  
        d['simplification'] = []
        ann = client.annotate(d['doc'])
        for sentence in ann.sentence:
            for triple in sentence.openieTriple:
                d['simplification'].append(
                    {
                        'prompt':"{} "+triple.relation,
                        "input":triple.subject + " " + triple.relation,
                        "target":" "+triple.object,
                        "subject":triple.subject,
                    }
                )
        result_data.append(d)

with open(path+".stanford_openie",'w') as f:
    json.dump(result_data,f,ensure_ascii=False,indent=2)
# # Compare the original triple with the processed triple
# for ori, proc in zip(preprocessed_triples,processed_triples):
#     print(f"Original : {ori['sub']} --- {ori['rel']} --> {ori['obj']}")
#     print(f"Processed: {proc['sub']} --- {proc['rel']} --> {proc['obj']}") 
#     print("")
