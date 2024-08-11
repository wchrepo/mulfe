
import os
import sys,json
# root_dir = os.path.dirname(os.getcwd())
# sys.path.append(root_dir)
# os.chdir(root_dir)
#%%
# import asyncio
# import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import openai
import argparse
import chatgen_strategies
parser = argparse.ArgumentParser(description='Call OpenAI API')
parser.add_argument('--path', type=str,
                    help='path to save the generated results')
parser.add_argument("--source_path",type=str,default=None,help="path to source data")
parser.add_argument('--max_retry', type=int, default=3,
                    help='max retry times')
parser.add_argument('--num_generate', type=int, default=500,
                    help='number of prompts to generate')
parser.add_argument('--num_workers',type=int,default=10)
parser.add_argument('--num_return',type=int,default=1)
parser.add_argument('--check_interval',type=float,default=0.1)
parser.add_argument('--submit_interval',type=float,default=1)#提交任务间隔，防止提交任务过快。
parser.add_argument('--random_seed', type=int, default=0,
                    help='random seed for generating prompts')
parser.add_argument('--strategy', type=str)
parser.add_argument("--eval",default=None)

# args,unparsed = parser.parse_known_args()
args = parser.parse_args()
path = args.path

strategy_kwargs = {}
if args.source_path:
    strategy_kwargs["source_path"] = args.source_path

import time,os,random,json,pickle,lzma
from tqdm.auto import tqdm

if not os.path.exists(f"{path}.state.pkl"):
    f = open(f"{path}.result.jsonl",'w')
    metaf = lzma.open(f"{path}.meta.jsonl.xz",'wt')
    state_dict = {}
else:
    f = open(f"{path}.result.jsonl",'a')
    metaf = lzma.open(f"{path}.meta.jsonl.xz",'at')
    with open(f"{path}.state.pkl",'rb') as statef:
        state_dict = pickle.load(statef)


strategy = state_dict.get("strategy",args.strategy)
prompt_init, prompt_get_iter = chatgen_strategies.get(strategy)
prompt_kwargs = state_dict.get('prompt_kwargs') or prompt_init(**strategy_kwargs)
start = state_dict.get('start', 0)
max_retry = state_dict.get('max_retry',args.max_retry)
num_generate = state_dict.get('num_generate',args.num_generate)
random_seed = state_dict.get('random_seed', args.random_seed)
random_state = state_dict.get('random_state', random.getstate())

if strategy.endswith("_json"):
    api_kwargs = dict(
        # model="gpt-3.5-turbo-1106",
        model = "gpt-4-1106-preview",
        top_p=0.99,
        n=args.num_return,
        response_format={ "type": "json_object" },
    )
else:
    api_kwargs = dict(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4-1106-preview",
        top_p=0.99,
        n=args.num_return,
    )

if args.eval:
    eval(args.eval)

random.seed(random_seed)
random.setstate(random_state)
iter_prompts = prompt_get_iter(**prompt_kwargs)


def call_openai(messages,postprocess):
    retry = 0
    while retry < max_retry:
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                **api_kwargs
                )
        except Exception as e:
            tqdm.write(str(e))
            if retry>max_retry:
                raise e
            time.sleep(20) #random.randint(5,15)
            retry += 1
            continue
        results, should_retry, should_write = postprocess(response)
        if should_retry:
            retry += 1
            continue
        break
    return messages,response,results,should_write
        


import atexit
@atexit.register
def handle_exit():
    print("Trying to exit gracefully") 
    try:
        wait_cache_finish()
    finally:
        dump_state()
        print("save state before exit")
        f.close()
        metaf.close()
    
def dump_state():
    with open(f"{path}.state.pkl",'wb') as statef:
        state_dict = {
            'strategy':strategy,
            'prompt_kwargs': prompt_kwargs, 
            'start': step,
            'max_retry':max_retry, 
            'num_generate': num_generate,
            'random_seed':random_seed,
            'random_state':random_state,
        }
        pickle.dump(state_dict,statef)



cache = []

def check_cache(): 
    global cache
    for i,c in enumerate(cache):
        if c.done():
            try:
                (messages,response,results,should_write)=c.result()
                metaf.write(json.dumps({"should_write":should_write,"messages":messages,"response":response},ensure_ascii=False)+'\n')
                if should_write:
                    for result in results:
                        f.write(result)
                    f.flush()
            except Exception as e:
                raise e
            finally:
                del cache[i]
                break

def wait_cache_finish():
    while len(cache)>0:
        check_cache()

cache = []
# lock = asyncio.Lock()
with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
    stopped = False
    for step in tqdm(range(start,num_generate)):
        while True:
            if len(cache)<args.num_workers:
                try:
                    messages,postprocess = next(iter_prompts)
                except StopIteration:
                    stopped = True
                    wait_cache_finish()
                    break
                cache.append(pool.submit(call_openai,messages,postprocess))
                break
            check_cache()
            time.sleep(args.check_interval)
        # dump_state()
        if stopped:
            break
        time.sleep(args.submit_interval)
    wait_cache_finish()

