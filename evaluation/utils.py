
import typing 
def tokenize_prefix_and_target(tokenizer,prefixes,targets,max_length=None):
    #因为不一定能用上return_offsets_mapping,这里直接通过两次tokenize判断prefix长度。
    #警告！需要处理一下padding left的情况！
    #警告！需要add_special_tokens = False，防止添加额外的token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
            labels[i,start] = -100 #让第一个token的label为-100，因为没有token能用来预测第一个token，right padding情况下可以不担心，但是left padding时不设置会导致padding用来预测第一个token。
        tokenized['labels']=labels
    else:
        tokenized = tokenizer(targets,
            return_tensors='pt',
            padding=True,truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        if tokenizer.padding_side == 'left':
            starts = tokenized.attention_mask.size(-1) - tokenized.attention_mask.sum(-1)
        else:
            starts = [0]*len(targets)
        labels = tokenized.input_ids.detach().clone()
        labels[range(len(targets)),starts] = -100
        labels[labels==tokenizer.pad_token_id]=-100
        tokenized['labels']=labels
    return tokenized

def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output