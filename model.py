from dataclasses import dataclass
import json
import os

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List

from transformers import AutoModel, BatchEncoding, PreTrainedModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, AutoModelForCausalLM
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, BitsAndBytesConfig)
from transformers.modeling_outputs import ModelOutput

from torch import cuda, bfloat16


import logging

logger = logging.getLogger(__name__)

from peft import PeftModel, prepare_model_for_kbit_training
from arguments import ModelArguments, DataArguments, PrivateTrainingArguments, DenseTrainingArguments

class LinearClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768
    ):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, h: Tensor = None):
        if h is not None:
            return self.linear(h)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _classifier_path = os.path.join(ckpt_dir, 'classifier.pt')
            if os.path.exists(_classifier_path):
                logger.info(f'Loading Classifier from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'classifier.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Classifier from scratch")
        return

    def save_classifier(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'classifier.pt'))
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(self._config, f)


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, p: Tensor = None):
        if p is not None:
            return self.linear(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


@dataclass
class GaLMOutput(ModelOutput):
    q_reps: Tensor = None
    k_reps: Tensor = None
    loss: Tensor = None
    logits: Tensor = None
    target: Tensor = None

class GaLMModel(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: PrivateTrainingArguments = None,

    ):
        super().__init__()
        self.lm = lm
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args

    def encode(self, query, keys, negs=None):
        if query is None or keys is None:
            return None
        inputs = {}
        for key in query.keys():
            inputs[key] = torch.cat([query[key], keys[key], negs[key]], dim=0) if negs is not None else torch.cat([query[key], keys[key]], dim=0)
    
        if "bert" in self.model_args.model_type:
            hidden = self.lm(**inputs).last_hidden_state
            reps = hidden[:, 0]
        elif "llama" in self.model_args.model_type:
            hidden = self.lm(**inputs, output_hidden_states=True)['hidden_states'][-1]
            reps = hidden[torch.arange(hidden.size(0)), inputs['attention_mask'].sum(dim=1) - 1]
        else:
            raise NotImplementedError
        return reps if negs is not None else torch.tensor_split(reps, 2)
    
    def encode_single(self, inputs):
        if inputs is None:
            return None
        if "bert" in self.model_args.model_type:
            hidden = self.lm(**inputs).last_hidden_state
            if self.pooler is not None:
                reps = self.pooler(p=hidden[:, 0])
            else:
                reps = hidden[:, 0]
        elif "llama" in self.model_args.model_type:
            hidden = self.lm(**inputs, output_hidden_states=True)['hidden_states'][-1]
            reps = hidden[torch.arange(hidden.size(0)), inputs['attention_mask'].sum(dim=1) - 1]
        else:
            raise NotImplementedError
        return reps

    def forward(self, query, key, neg_key=None):
        q_reps = self.encode_single(query)
        k_reps = self.encode_single(key)

        logits = torch.matmul(q_reps, k_reps.transpose(0, 1))
        target = torch.arange(
            logits.size(0),
            device=logits.device,
            dtype=torch.long
        )
        loss = self.cross_entropy(logits, target)

        return GaLMOutput(
            loss=loss,
            logits=logits,
            target=target,
            q_reps=q_reps.contiguous(),
            k_reps=k_reps.contiguous()
        )

    @classmethod
    def build(
        cls,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: PrivateTrainingArguments = None,
        **hf_kwargs,
    ):
        #load model
        if "bert" in model_args.model_type:
            if 'lora' in model_args.model_name_or_path:
                base_model = AutoModel.from_pretrained(model_args.config_name, **hf_kwargs)
                peft_model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path)
                lm = peft_model.merge_and_unload()
                if train_args.do_train:
                    trainable_layers = [lm.encoder.layer[-8], lm.pooler]
                    trainable_params = 0
                    for layer in trainable_layers:
                        for p in layer.parameters():
                            p.requires_grad = True
                            trainable_params += p.numel()
                    print(f"Trainable parameters count: {trainable_params}")
            else:
                lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                total_params = 0
                trainable_params = 0

                for p in lm.parameters():
                    p.requires_grad = False
                    total_params += p.numel()

                trainable_layers = [lm.encoder.layer[-8:], lm.pooler]

                for layer in trainable_layers:
                    for p in layer.parameters():
                        p.requires_grad = True
                        trainable_params += p.numel()
                print(f"Total parameters count: {total_params}")
                print(f"Trainable parameters count: {trainable_params}")
        elif "llama2" in model_args.model_type or "mistral" in model_args.model_type:
            if train_args.quantization:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True,)
                lm = LlamaForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=model_args.model_name_or_path,
                        quantization_config=bnb_config,
                        use_cache=False,
                        device_map="auto",
                        offload_folder="offload",
                    )
                # Prepare the model for int8 training if quantization is enabled
                lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=False)
            else:
                if 'lora' in model_args.model_name_or_path or 'nondp' in model_args.model_name_or_path or 'edge_flip' in model_args.model_name_or_path:
                    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto",offload_folder="offload",) 
                    model = PeftModel.from_pretrained(model=base_model, model_id=model_args.model_name_or_path,  device_map="auto",)
                    if train_args.resume_training:
                        print(f'Resuming from {model_args.model_name_or_path}')
                        lm = PeftModel.from_pretrained(model=base_model, model_id=model_args.model_name_or_path,  device_map="auto", is_trainable=True)
                    else:
                        model = PeftModel.from_pretrained(model=base_model, model_id=model_args.model_name_or_path,  device_map="auto",)
                        lm = model.merge_and_unload(safe_merge=True)
                else:
                    lm = LlamaForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=model_args.model_name_or_path,
                        device_map="auto",
                        offload_folder="offload",
                    )
        else:
            raise NotImplementedError
        
        model = cls(
            lm=lm,
            pooler=None,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model
    
    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)


class DPGaLMModel(GaLMModel):
    def forward(self, query, key, neg_key=None):
        if neg_key is not None:
            batch_size = len(query['input_ids'])
            reps = self.encode(query, key, neg_key)
            q_reps, k_reps, n_reps = reps[:batch_size], reps[batch_size:2*batch_size], reps[2*batch_size:]
        else:
            q_reps, k_reps = self.encode(query, key)

        if self.training:
            pos_scores = (q_reps * k_reps).sum(dim=1, keepdim=True)
            neg_scores = torch.einsum('ijk,jk->ji', n_reps.view(-1,batch_size,q_reps.size(-1)), q_reps)
            logits = torch.cat([pos_scores, neg_scores], dim=1)
            target = torch.zeros(pos_scores.size(0), dtype=torch.long, device=logits.device)

            loss = self.cross_entropy(logits, target)
        else:
            logits = torch.matmul(q_reps, k_reps.transpose(0, 1))
            target = torch.arange(
                logits.size(0),
                device=logits.device,
                dtype=torch.long
            )
            loss = self.cross_entropy(logits, target)

        return GaLMOutput(
            loss=loss,
            logits=logits,
            target=target,
            q_reps=q_reps.contiguous(),
            k_reps=k_reps.contiguous()
        )
    
class GaLMModelforNCC(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            pooler: nn.Module = None,
            classifier: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: DenseTrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.pooler = pooler
        self.classifier = classifier

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args


    def forward(
            self,
            query: Dict[str, Tensor] = None,
            keys: Tensor = None,
    ):

        q_hidden, q_reps = self.encode(query)
        labels = keys

        if q_reps is None:
            return GaLMOutput(
                q_reps=q_reps,
            )

        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)

        scores = self.classifier(q_reps)

        loss = self.cross_entropy(scores, labels)

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction

        return GaLMOutput(
            loss=loss,
            logits=scores,
            target=labels,
            q_reps=q_reps,
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        if 'bert' in self.model_args.model_type:
            psg_out = self.lm(**psg['center_input'])
            try:
                p_hidden = psg_out.last_hidden_state
                if self.pooler is not None:
                    p_reps = self.pooler(p=p_hidden)  # D * d
                else:
                    p_reps = p_hidden[:, 0]
            except:
                p_reps = psg_out
                p_hidden = None
        elif self.model_args.model_type == "llama2":
            p_hidden = self.lm(**psg['center_input'], output_hidden_states=True)['hidden_states'][-1]
            p_reps = p_hidden[torch.arange(p_hidden.size(0)), psg['center_input']['attention_mask'].sum(dim=1) - 1]
        else:
            psg_out = self.lm(**psg)
                
        return p_hidden, p_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def build_classifier(data_args, model_args):
        classifier = LinearClassifier(
            model_args.projection_in_dim,
            data_args.class_num
        )
        ckpt = model_args.model_name_or_path #if not model_args.adapt_domain else None
        classifier.load(ckpt)
        return classifier

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: DenseTrainingArguments,
            **hf_kwargs,
    ):  
        # load model
        if 'bert' in model_args.model_type:
            if 'lora' in model_args.model_name_or_path:
                base_model = AutoModel.from_pretrained(model_args.config_name, **hf_kwargs)
                peft_model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path)
                lm = peft_model.merge_and_unload()
            else:
                lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            if 'large' in model_args.tokenizer_name:
                model_args.projection_in_dim = 1024
        elif model_args.model_type == "llama2":
            print(f'Loading from {model_args.model_name_or_path}')
            if 'lora' in model_args.model_name_or_path or 'nondp' in model_args.model_name_or_path:
                base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", offload_folder="offload",) 
                model = PeftModel.from_pretrained(model=base_model, model_id=model_args.model_name_or_path,  device_map="auto",)
                lm = model.merge_and_unload(safe_merge=True)
            else:
                lm = LlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_args.model_name_or_path,
                    device_map="auto",
                    offload_folder="offload",
                )
            model_args.projection_in_dim = 4096
        else:
            raise NotImplementedError
        
        # init classifier
        classifier = cls.build_classifier(data_args, model_args)

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm=lm,
            pooler=pooler,
            classifier=classifier,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        # self.lm.save_pretrained(output_dir)
        self.classifier.save_classifier(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)
