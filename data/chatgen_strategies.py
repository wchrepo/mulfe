#%%
import os
import sys,json,random
# root_dir = os.getcwd().removesuffix("/llm_gen")
# sys.path.append(root_dir)
# os.chdir(root_dir)
import functools



#%% Example Pools take from GKP

clozes_prompt = [
{
"role":"system",
"content":"""Create clozes based on the text, using the underline "___" to mask entity or property phrases. Note that the clozes should be unambiguous and answerable without the context, and you should provide the answer after the clozes. Example:
```
Danielle Yvonne Marie Antoinette Darrieux (1 May 1917 – 17 October 2017) was a French actress of stage, television and film, as well as a singer and dancer.

1. Danielle Darrieux was ___. 
(Answer: a French actress)

2. Danielle Darrieux was born on ___.
(Answer: 1 May 1917)

...
```
"""
},
{
"role":"user",
"content":"""Now create clozes based the text:

{doc}
"""
}
]

# clozes_prompt_no_example = [
# {
# "role":"system",
# "content":"""Create clozes based on the text, using the underline "___" as the blank. Note that you should provide the answer after the clozes. You should follow this format:
# ```
# 1. cloze 1
# (Answer: answer 1)

# 2. cloze 2
# (Answer: answer 2) 

# ...
# ```
# """
# },
# {
# "role":"user",
# "content":"""Now create clozes based the text:

# {doc}
# """
# }
# ]

clozes_prompt_no_example = [
{
"role":"system",
"content":"""Create clozes based on the text, using the underline "___" to mask entity or property phrases. Note that the clozes should be unambiguous and answerable without the context, and you should provide the answer after the clozes. You should follow this format:
```
1. cloze 1
(Answer: answer 1)

2. cloze 2
(Answer: answer 2) 

...
```
"""
},
{
"role":"user",
"content":"""Now create {num_clozes} clozes based the text:

{doc}
"""
}
]

"""Rewrite the clozes into wh-questions. Example:
"""

clozes_json_example = [
{
"role":"system",
"content":"""Based on the given text, create a list of clozes (using the underline "___" as the mask) of different difficulty levels. Level 1 should be based on some exact fragments of the source text. Level 2 include simple synonymous variants or paraphrases of the original text. Level 3 requires some reasoning or summarizing processes based on the original text. Note that the clozes should be unambiguous and answerable without the context. You should also provide the correct answer as well as specific tags to indicate the questiontype. The answer MUST be short phrases rather than a full sentence. Your response should follow this JSON format.
```
{{"probes":[
    {{
        "query": "...", # A cloze
        "answer": "...", # The correct answer
        "level":"1", # Difficulty level: 1, 2, 3
        "tag":["..."]
    }},
    {{
        ... # More instances
    }}
]}}
```
"""
},
{
"role":"user",
"content":"""Create 6 clozes and questions based on the text:

January 2, 2022 – Abdalla Hamdok resigns as Prime Minister of Sudan amid deadly protests.
"""
},
{
"role":"assistant",
"content":"""{{"probes":[
    {{
        "query": "Abdalla Hamdok resigns as ___ of Sudan amid deadly protests.",
        "answer": "Prime Minister",
        "level":"1",
        "tag":["Reciting"]
    }},
    {{
        "query": "Abdalla Hamdok resigns as Prime Minister of ___ amid deadly protests.",
        "answer": "Sudan",
        "level":"1",
        "tag":["Reciting"]
    }},
    {{
        "query": "Abdalla Hamdok is no longer as ___ of Sudan.",
        "answer": "Prime Minister",
        "level":"2",
        "tag":["Rephrasing"]
    }},
    {{
        "query": "___, who served as Prime Minister of Sudan, resigned amid deadly protests.",
        "answer": "Abdalla Hamdok",
        "level":"2",
        "tag":["Rephrasing"]
    }},
    {{
        "query": "Abdalla Hamdok is former ___ of Sudan.",
        "answer": "Prime Minister",
        "level":"3",
        "tag":["Status Inference"]
    }},
    {{
        "query": "Is Abdalla Hamdok a successful politican? ___ (Yes/No)",
        "answer": "No",
        "level":"3",
        "tag":["Public Opinion Inference"]
    }}
]}}"""
},
{
"role":"user",
"content":"""Create {num_clozes} clozes or questions based on the text:

{doc}
"""
}
]

#such as Reciting, Rephrasing, Counting, etc. 
clozes_qa_mixed_json_example = [
{
"role":"system",
"content":"""Based on the given text, create a list of clozes (using the underline "___" as the mask) or questions of different difficulty levels. Level 1 should be based on some exact fragments of the source text. Level 2 include simple synonymous variants or paraphrases of the original text. Level 3 requires some reasoning or summarizing processes based on the original text. Note that the clozes or questions should be unambiguous and answerable without the context. You should also provide the correct answer as well as specific tags to indicate the question type. The answer MUST be short phrases rather than a full sentence. Your response should follow this JSON format.
```
{{"probes":[
    {{
        "query": "...", # A cloze or question
        "answer": "...", # The correct answer
        "level":"1", # Difficulty level: 1, 2, 3
        "tag":["..."]
    }},
    {{
        ... # More instances
    }}
]}}
```
"""
},
{
"role":"user",
"content":"""Create 6 clozes and questions based on the text:

January 2, 2022 – Abdalla Hamdok resigns as Prime Minister of Sudan amid deadly protests.
"""
},
{
"role":"assistant",
"content":"""{{"probes":[
    {{
        "query": "Abdalla Hamdok resigns as ___ of Sudan amid deadly protests.",
        "answer": "Prime Minister",
        "level":"1",
        "tag":["Reciting"]
    }},
    {{
        "query": "Abdalla Hamdok resigns as Prime Minister of ___ amid deadly protests.",
        "answer": "Sudan",
        "level":"1",
        "tag":["Reciting"]
    }},
    {{
        "query": "Abdalla Hamdok is no longer as ___ of Sudan.",
        "answer": "Prime Minister",
        "level":"2",
        "tag":["Rephrasing"]
    }},
    {{
        "query": "Who served as Prime Minister of Sudan and resigned amid deadly protests on January 2, 2022?",
        "answer": "Abdalla Hamdok",
        "level":"2",
        "tag":["Rephrasing"]
    }},
    {{
        "query": "Abdalla Hamdok is former ___ of Sudan.",
        "answer": "Prime Minister",
        "level":"3",
        "tag":["Status Inference"]
    }},
    {{
        "query": "Is Abdalla Hamdok a successful politican?",
        "answer": "No",
        "level":"3",
        "tag":["Public Opinion Inference"]
    }}
]}}"""
},
{
"role":"user",
"content":"""Create {num_clozes} clozes or questions based on the text:

{doc}
"""
}
]

#%%
def prompt_fill(template,**ctx):
    prompt = [{"role":t['role'],"content":t['content'].format(**ctx)} for t in template]
    return prompt

#%% Patterns
import re

# results_pattern = re.compile(
#     r"\d+\. (?P<query>.+)\n?\(Answer: (?P<answer>.+)\)"
# )


results_pattern = re.compile(
    r"\d+\. (?P<query>.*_.*)\n?\(Answer: (?P<answer>.+)\)"
)

#%% postprocess

def postprocess_qa(response):
    """
    return List[result], should_retry, should_write
    """
    results = []
    for r in response['choices']:
        matches = [match.groupdict() for match in results_pattern.finditer(r['message']['content'])]
        results.extend(matches)
    if results:
        return results,False,True
    else:
        return results,True,False
    
def postprocess_qa_json(response):
    """
    return List[result], should_retry, should_write
    """
    results = []
    for r in response['choices']:
        try:
            matches = json.loads(r['message']['content'])['probes']
            results.extend(matches)
        except:
            pass        
    if results:
        return results,False,True
    else:
        return results,True,False

def get_postprocess_qa(d,postprocess=postprocess_qa):
    def postprocess_fn(response):
        results,should_response,should_write = postprocess(response)
        result_dict = d.copy()
        result_dict['generated_probes'] = results
        return json.dumps(result_dict,ensure_ascii=False)+'\n',should_response,should_write
    return postprocess_fn

def get_prompt(data,prompt=clozes_prompt_no_example,postprocess=postprocess_qa,**kwargs):
    while len(data)>0:
        d = data.pop(0)
        # prompt = prompt_fill(clozes_prompt,d)
        filled_prompt = prompt_fill(prompt,**d,**kwargs)
        yield filled_prompt,get_postprocess_qa(d,postprocess=postprocess)


def init_params(source_path="stage1/dune_new_info.json",**extra_kwargs):
    kwargs = {**extra_kwargs}
    with open(source_path) as f:
        kwargs['data'] = json.load(f)
    
    return kwargs

    
    
# %%
strategies = {
    "dune_clozes":(functools.partial(init_params,num_clozes="several"),get_prompt),
    "dune_clozes_8":(functools.partial(init_params,num_clozes="8"),get_prompt),
    "dune_clozes_10":(functools.partial(init_params,num_clozes="10"),get_prompt),
    "dune_clozes_3":(functools.partial(init_params,num_clozes="3"),get_prompt),
    "dune_clozes_ex":(functools.partial(init_params,num_clozes="several"),functools.partial(get_prompt,prompt=clozes_prompt)),
    "dune_clozes_json":(functools.partial(init_params,num_clozes="several"),functools.partial(get_prompt,prompt=clozes_json_example,postprocess=postprocess_qa_json)),
    "dune_clozes_qa_json":(functools.partial(init_params,num_clozes="several"),functools.partial(get_prompt,prompt=clozes_qa_mixed_json_example,postprocess=postprocess_qa_json)),
    "ecbd2022_clozes_json":(functools.partial(init_params,source_path="stage1/ecbd_2022.json",num_clozes="several"),functools.partial(get_prompt,prompt=clozes_json_example,postprocess=postprocess_qa_json)),
    "ecbd2022_clozes_qa_json":(functools.partial(init_params,source_path="stage1/ecbd_2022.json",num_clozes="several"),functools.partial(get_prompt,prompt=clozes_qa_mixed_json_example,postprocess=postprocess_qa_json)),
    "entityinference_clozes_json":(functools.partial(init_params,source_path="stage1/entity_inferences.json",num_clozes="several"),functools.partial(get_prompt,prompt=clozes_json_example,postprocess=postprocess_qa_json)),
    "entityinference_clozes_qa_json":(functools.partial(init_params,source_path="stage1/entity_inferences.json",num_clozes="several"),functools.partial(get_prompt,prompt=clozes_qa_mixed_json_example,postprocess=postprocess_qa_json)),
    "ecbd2023_clozes_8":(functools.partial(init_params,source_path="stage1/ecbd_2023.json",num_clozes="8"),get_prompt),
    "ecbd2022_clozes_8":(functools.partial(init_params,source_path="stage1/ecbd_2022.json",num_clozes="8"),get_prompt),
}

get = strategies.get

#%%
if __name__ == "__main__":
    import openai
    kwargs = init_params()
    prompt_iter=get_prompt(**kwargs)
    messages,postprocess = next(prompt_iter)
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                top_p=0.95
                )
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                top_p=0.95,
                response_format={ "type": "json_object" },
                )