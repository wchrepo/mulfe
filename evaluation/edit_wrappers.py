

import contextlib
import torch
import re
from utils import tokenize_prefix_and_target
from peft import LoraConfig,get_peft_model
from baselines.mend.algs.mend_rawdoc import MEND_RAWDOC
from baselines.mend.algs.mend_augdoc import MEND_AUGDOC
from baselines.mend.algs.mend import MEND
from baselines.memit.memit_main import apply_memit_to_model
from baselines.memit.rome.rome_main import apply_rome_to_model
from copy import deepcopy
import transformers
import hydra
import os
from omegaconf import OmegaConf

class EditWrapper:
    def __init__(self,model,tokenizer,config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    def edit(self,edits):
        #edits: list of edits
        raise NotImplementedError

    def autorestore(self):
        raise NotImplementedError

class NoEdit(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)

    def edit(self,edits):
        #do nothing
        return {}
    @contextlib.contextmanager
    def autorestore(self):
        try:
            yield {}
        finally:
            pass

class FineTuning(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        if hasattr(self.config.editor,"trainable_pattern") and \
            "{layers}" in self.config.editor.trainable_pattern:
            if self.config.editor.layers is not None:
                layers = []
                for l in str(config.editor.layers).split(","):
                    if "-" in l:
                        start,_,end = l.split("-")
                        layers += [str(i) for i in range(int(start),int(end)+1)]
                    else:
                        layers.append(l)
                layers = "("+"|".join(layers)+")"
            else:
                layers = "."
            self.config.editor.trainable_pattern = \
                config.editor.trainable_pattern.replace("{layers}",layers)

    def find_trainable_parameters(self):
        self.model.requires_grad_(False)
        trainable = {}
        for n, p in self.model.named_parameters():
            if re.search(self.config.editor.trainable_pattern,n):
                p.requires_grad_(True)
                trainable[n] = p
        return trainable
    
    @contextlib.contextmanager
    def autorestore(self):
        state_backup = self.find_trainable_parameters()
        if self.config.low_vram:
            state_backup = {k:v.detach().clone().to(device='cpu') for k,v in state_backup.items()}
        else:
            state_backup = {k:v.detach().clone() for k,v in state_backup.items()}
        try:
            yield state_backup
        finally:
            self.model.load_state_dict(state_backup,strict=False)
    
    def collate_fn(self,edits):
        prefixes = []
        targets = []
        for e in edits:
            if "doc" in e:
                prefixes.append("")
                targets.append(e['doc'])
            elif 'input' in e:
                prefixes.append(e['input'])
                targets.append(e['target'])
        return tokenize_prefix_and_target(self.tokenizer,prefixes,targets)
    
    def ict_distill_collate_fn(self,edits):
        if self.config.editor.edit_loss_coeff > 0.:
            edit_batch = tokenize_prefix_and_target(self.tokenizer,[],[e['doc'] for e in edits ])
        else:
            edit_batch = None
            # loss = loss + doc_loss
        if self.config.editor.aug_loss_coeff > 0:
            prefixes = [p['input'] for e in edits for p in e['simplification']]
            targets = [p['target'] for e in edits for p in e['simplification']]
            if targets:
                ict_prefixes = [e['doc']+"\n\n"+p['input'] for e in edits for p in e['simplification']]
                ict_batch = tokenize_prefix_and_target(self.tokenizer,ict_prefixes,targets)
                aug_batch = tokenize_prefix_and_target(self.tokenizer,prefixes,targets)
                return edit_batch, aug_batch, ict_batch
        return edit_batch, None, None
        

    def cut_minibatch(self,batch,max_minibatch_size=None):
        #To avoid OOM, make sure the tokens amounts < config.editor.minibatch_tokens 
        if max_minibatch_size is None and self.config.editor.minibatch_tokens > 0:
            input_length = batch['input_ids'].size(1)
            max_minibatch_size = max(self.config.editor.minibatch_tokens // input_length,1)
        if max_minibatch_size is not None:
            minibatches = [transformers.BatchEncoding(
                {k:v[i:i+max_minibatch_size] for k,v in batch.items()}) 
                for i in range(0,batch['input_ids'].size(0),max_minibatch_size)]
            return minibatches
        else:
            return [batch]


    def edit(self,edits):
        if self.config.editor.train_mode:
            self.model.train()
        else:
            self.model.eval()
        torch.cuda.empty_cache()
        trainable_parameters = self.find_trainable_parameters()
        # for n, p in trainable_parameters.items():
        #     p.requires_grad_(True)
        optimizer = torch.optim.__dict__.get(self.config.editor.opt_name)(
            trainable_parameters.values(),
            **dict(self.config.editor.opt_kwargs)
        )
        if getattr(self.config.editor,"ict_distill",False):
            edit_batch, aug_batch, ict_batch = self.ict_distill_collate_fn(edits)
            if edit_batch is not None:
                all_edit_target_tokens = (edit_batch['labels'][:,1:]!=-100).sum().detach().item()
                edit_minibatches = self.cut_minibatch(edit_batch)
            else:
                edit_minibatches = []
            if aug_batch is not None:
                all_aug_target_tokens = (aug_batch['labels'][:,1:]!=-100).sum().detach().item()
                ict_minibatches = self.cut_minibatch(ict_batch)
                aug_minibatches = self.cut_minibatch(aug_batch,max_minibatch_size=ict_minibatches[0].input_ids.size(0))
            else:
                aug_minibatches = []
                ict_minibatches = []
            ict_outputs_for_minibatches = []
            for aug_minibatch,ict_minibatch in zip(aug_minibatches,ict_minibatches):
                aug_minibatch = aug_minibatch.to(self.model.device)
                ict_minibatch = ict_minibatch.to(self.model.device)
                with torch.no_grad():    
                    ict_outputs = self.model(**ict_minibatch).logits[:,:-1] [ict_minibatch['labels'][:,1:]!=-100]
                    aug_outputs = self.model(**aug_minibatch).logits[:,:-1] [aug_minibatch['labels'][:,1:]!=-100]
                    ict_outputs = ict_outputs + self.config.editor.ict_contra_coeff*(ict_outputs-aug_outputs)
                    ict_outputs = ict_outputs.detach()
                    ict_outputs_for_minibatches.append(ict_outputs)

            for _ in range(self.config.editor.steps):
                optimizer.zero_grad()
                loss = 0.
                for minibatch in edit_minibatches:
                    #grad accumuluation
                    target_tokens = (minibatch['labels'][:,1:]!=-100).sum().detach().item()
                    miniloss = self.config.editor.edit_loss_coeff*self.model(**minibatch.to(device=self.model.device)).loss*(target_tokens/all_edit_target_tokens)
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                for aug_minibatch,ict_minibatch,ict_outputs in zip(aug_minibatches,ict_minibatches,ict_outputs_for_minibatches):
                    aug_minibatch = aug_minibatch.to(self.model.device)
                    target_tokens = (aug_minibatch["labels"][:,1:]!=-100).sum().detach().item()
                    aug_results = self.model(**aug_minibatch)
                    target_loss = aug_results.loss
                    aug_outputs = aug_results.logits[:,:-1] [aug_minibatch['labels'][:,1:]!=-100]
                    kl_loss = (
                        ict_outputs.softmax(-1) * (ict_outputs.log_softmax(-1) - aug_outputs.log_softmax(-1))
                    ).sum(-1).mean()
                    miniloss = self.config.editor.aug_loss_coeff*( 
                        (self.config.editor.kl_coeff)*kl_loss + (1-self.config.editor.kl_coeff)*target_loss) * target_tokens/all_aug_target_tokens
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                if loss < self.config.editor.early_stop:
                    optimizer.zero_grad()
                    break
                optimizer.step()
        else:
            batch = self.collate_fn(edits)
            all_target_tokens = (batch['labels']!=-100).sum().detach().item()
            minibatches = self.cut_minibatch(batch)
            for _ in range(self.config.editor.steps):
                optimizer.zero_grad()
                loss = 0.
                for minibatch in minibatches:
                    #grad accumuluation
                    target_tokens = (minibatch['labels']!=-100).sum().detach().item()
                    miniloss = self.model(**minibatch.to(device=self.model.device)).loss*(target_tokens/all_target_tokens)
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                if loss < self.config.editor.early_stop:
                    optimizer.zero_grad()
                    break
                optimizer.step()
        return edits

class LoRAFineTuning(FineTuning):
    def __init__(self,model,tokenizer,config):
        if hasattr(config.editor,"layers") and config.editor.layers is not None:
            layers = []
            for l in str(config.editor.layers).split(","):
                if "-" in l:
                    start,_,end = l.split("-")
                    layers += list([range(int(start),int(end)+1)])
                else:
                    layers.append(int(l))
            config.editor.lora_kwargs.layers_to_transform = layers
        lora_config = LoraConfig(task_type="CAUSAL_LM",**config.editor.lora_kwargs)
        model = get_peft_model(model,lora_config)
        super().__init__(model,tokenizer,config)

    def find_trainable_parameters(self):
        trainable = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                trainable[n] = p
        return trainable
    
class MENDWrapper(EditWrapper):
    #Modified from Meng 2022
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        self.original_model = model
        self.init_model(model,tokenizer,config.editor.hparams)
    

    def init_model(self, model, tok, params):
        train_ds = (
            "counterfact-" if params.counterfact else ("zsre-" if params.zsre else "")
        )
        mini_string = "mini-" if params.mini else ""

        if params.model_name == "gpt2-xl":
            model_name = "gpt2-xl"
            modelcode = "gpt2xl"
        elif params.model_name == "EleutherAI/gpt-j-6B":
            model_name = "gpt-j-6b"
            modelcode = "gptj"
        elif params.model_name == "EleutherAI/gpt-neo-2.7B":
            model_name = "gpt-neo-2.7b"
            modelcode = "gptneo27"
        else:
            raise NotImplementedError
        if hasattr(self.config.editor,"model_filename"):
            model_filename = self.config.editor.model_filename
        else:
            model_filename = (
                f"mend-{mini_string}{params.n_toks}tok-{train_ds}{model_name}.pt"
            )
        model_dir = self.config.editor.model_dir

        os.makedirs(model_dir, exist_ok=True)
        if not os.path.isfile(f"{model_dir}/{model_filename}"):
            remote_url = f"https://memit.baulab.info/data/weights/{model_filename}"
            print(f"Attemping to download from {remote_url}")
            torch.hub.download_url_to_file(remote_url, f"{model_dir}/{model_filename}")
        with hydra.initialize(config_path="baselines/mend/config", job_name="run"):
            config = hydra.compose(
                config_name="config",
                overrides=[
                    "+alg=mend",
                    "+experiment=gen",
                    f"+model={modelcode}",
                    f"data.path=data/{params.n_toks}token/data/self_sample/",
                ],
            )

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND(self.model, config, lambda: deepcopy(self.original_model))
        d = torch.load(f"{model_dir}/{model_filename}")
        self.alg.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        self.alg.cuda()
        self.alg.train(False)

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in config.model.inner_params:
                p.requires_grad = False
        self.is_init = True

    def edit(
        self,
        edits
    ):
        # Define i/o
        # targets = [
        #     (" " if edit["target"][0] != " " else "")
        #     + edit["target"]
        #     for edit in edits
        # ]
        # sentences = [
        #     edit["input"] + targets[i]
        #     for i, edit in enumerate(edits)
        # ]

        # # Tokenize
        # assert self.tokenizer.padding_side == 'right'
        # sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
        #     "cuda"
        # )
        # target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(
        #     "cuda"
        # )

        # # Define labels
        # label_tok = deepcopy(sent_tok["input_ids"])
        # for i in range(label_tok.size(0)):
        #     target_len = target_tok["attention_mask"][i].sum()
        #     padding_len = (
        #         sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
        #     )
        #     label_tok[i][: -target_len - padding_len] = -100
        #     label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        # edit_inner = dict(
        #     input_ids=sent_tok["input_ids"],
        #     attention_mask=sent_tok["attention_mask"],
        #     labels=label_tok,
        # )

        prefixes = []
        targets = []
        for e in edits:
            if "doc" in e:
                prefixes.append("")
                targets.append(e['doc'])
            elif 'input' in e:
                prefixes.append(e['input'])
                targets.append(e['target'])
        edit_inner = tokenize_prefix_and_target(self.tokenizer,prefixes,targets).to(device=self.model.device)

        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        edited_model = self.alg.edit(edit_inner, return_edited_model=True)
        self.model = edited_model
        # return model, weights_copy
    @contextlib.contextmanager
    def autorestore(self):
        try:
            self.model = self.original_model
            yield self.model
        finally:
            del self.model
            self.model = self.original_model


class MENDDOCWrapper(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        self.original_model = model
        self.init_model(model,tokenizer)
    

    def init_model(self, model, tok):
        train_ds = (
            "mulfe"
        )

        if self.config.model_name == "gpt2-xl":
            model_name = "gpt2-xl"
            modelcode = "gpt2xl"
        elif self.config.model_name == "EleutherAI/gpt-j-6B":
            model_name = "gpt-j-6b"
            modelcode = "gptj"
        elif self.config.model_name == "EleutherAI/gpt-neo-2.7B":
            model_name = "gpt-neo-2.7b"
            modelcode = "gptneo27"
        if hasattr(self.config.editor,"model_filename"):
            model_filename = self.config.editor.model_filename
        else:
            model_filename = (
                f"menddoc--{train_ds}{model_name}.pt"
            )
        model_dir = self.config.editor.model_dir
        if os.path.exists(f"{model_dir}/{model_filename}.config"):
            config = OmegaConf.load(f"{model_dir}/{model_filename}.config")
        else:
            with hydra.initialize(config_path="baselines/mend/config", job_name="run"):
                config = hydra.compose(
                    config_name="config",
                    overrides=[
                        "+alg=mend_rawdoc",
                        "+experiment=mulfe_train",
                        f"+model={modelcode}",
                    ],
                )

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        # self.model = model
        # self.tokenizer = tok
        add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND_RAWDOC(self.model, config, lambda: deepcopy(self.original_model),tokenizer=self.tokenizer)
        d = torch.load(f"{model_dir}/{model_filename}")
        self.alg.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        self.alg.cuda()
        self.alg.train(False)

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in config.model.inner_params:
                p.requires_grad = False
        self.is_init = True

    def edit(
        self,
        edits
    ):
        # Tokenize
        assert self.tokenizer.padding_side == 'right'
        
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        edited_model = self.alg.edit([edit['doc'] for edit in edits], return_edited_model=True)
        self.model = edited_model
        # return model, weights_copy
    @contextlib.contextmanager
    def autorestore(self):
        try:
            self.model = self.original_model
            yield self.model
        finally:
            del self.model
            self.model = self.original_model

class MENDAUGWrapper(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        self.original_model = model
        self.init_model(model,tokenizer)
    

    def init_model(self, model, tok):
        train_ds = (
            "mulfe"
        )

        if self.config.model_name == "gpt2-xl":
            model_name = "gpt2-xl"
            modelcode = "gpt2xl"
        elif self.config.model_name == "EleutherAI/gpt-j-6B":
            model_name = "gpt-j-6b"
            modelcode = "gptj"
        elif self.config.model_name == "EleutherAI/gpt-neo-2.7B":
            model_name = "gpt-neo-2.7b"
            modelcode = "gptneo27"
        if hasattr(self.config.editor,"model_filename"):
            model_filename = self.config.editor.model_filename
        else:
            model_filename = (
                f"menddoc--{train_ds}{model_name}.pt"
            )
        model_dir = self.config.editor.model_dir
        if os.path.exists(f"{model_dir}/{model_filename}.config"):
            config = OmegaConf.load(f"{model_dir}/{model_filename}.config")
        else:
            with hydra.initialize(config_path="baselines/mend/config", job_name="run"):
                config = hydra.compose(
                    config_name="config",
                    overrides=[
                        "+alg=mend_augdoc",
                        "+experiment=mulfe_train_aug",
                        f"+model={modelcode}",
                    ],
                )

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        # self.model = model
        # self.tokenizer = tok
        add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND_AUGDOC(self.model, config, lambda: deepcopy(self.original_model),tokenizer=self.tokenizer)
        d = torch.load(f"{model_dir}/{model_filename}")
        self.alg.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        self.alg.cuda()
        self.alg.train(False)

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in config.model.inner_params:
                p.requires_grad = False
        self.is_init = True

    def edit(
        self,
        edits
    ):
        # Tokenize
        assert self.tokenizer.padding_side == 'right'
        inputs = []
        for e in edits:
            d = {"edit":e['doc']}
            if "simplification" in e:
                d['aug_pairs'] = e['simplification']
            inputs.append(d)
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        edited_model = self.alg.edit(inputs, return_edited_model=True)
        self.model = edited_model
        # return model, weights_copy
    @contextlib.contextmanager
    def autorestore(self):
        try:
            self.model = self.original_model
            yield self.model
        finally:
            del self.model
            self.model = self.original_model

class MEMITWrapper(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.original_weights = {}

    def edit(
        self,
        edits
    ):
        # translate edits to requests
        requests = [{"prompt":e['prompt'],"subject":e['subject'],"target_new":{"str":e['target']}} for e in edits]
        
        
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        edited_model,original_weights = apply_memit_to_model(
            self.model,self.tokenizer,requests,
            hparams=self.config.editor.hparams,return_orig_weights=True)
        if len(self.original_weights) == 0:
            self.original_weights = original_weights
        self.model = edited_model
        # return model, weights_copy
    @contextlib.contextmanager
    def autorestore(self):
        try:
            self.original_weights = {}
            yield 
        finally:
            self.model.load_state_dict(self.original_weights,strict=False)

class ROMEWrapper(EditWrapper):
    def __init__(self,model,tokenizer,config):
        super().__init__(model,tokenizer,config)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.original_weights = {}

    def edit(
        self,
        edits
    ):
        # translate edits to requests
        requests = [{"prompt":e['prompt'],"subject":e['subject'],"target_new":{"str":e['target']}} for e in edits]
        
        
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        edited_model,original_weights = apply_rome_to_model(
            self.model,self.tokenizer,requests,
            hparams=self.config.editor.hparams,return_orig_weights=True)
        if len(self.original_weights) == 0:
            self.original_weights = original_weights
        self.model = edited_model
        # return model, weights_copy
    @contextlib.contextmanager
    def autorestore(self):
        try:
            self.original_weights = {}
            yield 
        finally:
            self.model.load_state_dict(self.original_weights,strict=False)

    

class SimplificationWrapper(EditWrapper):

    def __init__(self,editor):
        self.editor = editor
        self.autorestore = editor.autorestore
    
    def __getattr__(self, name):
        return getattr(self.editor,name)
    def edit(self,edits):
        simplified = self.simplify(edits)
        if simplified:
            return self.editor.edit(simplified)
    
    def simplify(self,edits):
        #simplification should be done in preprocessing in evaluation
        return sum([e['simplification'] for e in edits],[])
    
class SimplificationWithRawDoc(SimplificationWrapper):

    def __init__(self,editor):
        super().__init__(editor)
    
    def simplify(self,edits):
        #simplification should be done in preprocessing in evaluation
        return sum([e['simplification'] for e in edits],[{"doc":e['doc']} for e in edits])
class OpenIESimplification(SimplificationWrapper):
    def __init__(self,editor):
        super().__init__(editor)
    def simplify(self,edits):
        return edits

class QGSimplification(SimplificationWrapper):
    def __init__(self,editor):
        super().__init__(editor)
    def simplify(self,edits):
        return edits
    
class AutoQGSimplification(SimplificationWrapper):
    def __init__(self,editor):
        super().__init__(editor)
    def simplify(self,edits):
        return edits