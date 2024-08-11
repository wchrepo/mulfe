import math
def summarize(results,metrics=['acc','ppl']):
    # all_scores = {}
    # for r in results:
    #     for m,v in r['metrics'].items():
    #         if m in metrics:
    #             all_scores.setdefault(m,[]).append(v)
    # outputs = {}
    # for k,v in all_scores.items():
    #     outputs[k] = sum(v)/len(v)
    # return outputs
    outputs = {}
    if len(results) == 0:
        return outputs
    if 'acc' in metrics:
        counts = [r['metrics']['acc'] for r in results if 'acc' in r['metrics']]
        if len(counts)>0:
            outputs['acc'] = sum(counts)/len(counts)
    if 'ppl' in metrics:
        nlls = []
        tokens = 0
        for r in results:
            if 'nll' in r['metrics']:
                nlls.append(r['metrics']['nll'])
                tokens += r['metrics']['tokens']
        if tokens>0:
            outputs['ppl'] = math.exp(sum(nlls)/tokens)
    return outputs


