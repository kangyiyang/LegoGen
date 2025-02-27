# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from ..model.llama_model import ModelArgs, Transformer
from ..model.tokenizer import brick_tokenizer
from ..model.clip import clip
from ..utils.utils import sample_logits, relative_transfer

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        text_model_size: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        position_dim: int = 128,
        out_pad_dim: int = 1,
        rank: int = 2,
        c_n_heads: int = 32,
        text_dim: int = 512,
        add_cross: int = 2,
        seed: int = 1,
        pretrain: bool = True,
    ):
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                position_dim=position_dim,
                out_pad_dim=out_pad_dim,
                rank=rank,
                c_n_heads=c_n_heads,
                # patch_h=patch_h,
                # patch_w=patch_w,
                text_dim=text_dim,
                add_cross=add_cross,
                **params,
            )
        brick_tokenizer_instance=brick_tokenizer()
        model_args.vocab_size = brick_tokenizer_instance.n_words
        print(f'The vocab of LEGO is {model_args.vocab_size}')

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_dtype(torch.half)
        # torch.set_default_device('cuda')

        if not pretrain:
            text_model, _ = clip.load(text_model_size, device='cuda')
        else:
            text_model = None
        model = Transformer(model_args, text_model)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model), checkpoint

    def __init__(self, model: Transformer):
        self.model = model
        self.brick_tokenizer=brick_tokenizer()
    
    @torch.inference_mode()
    def generate(
        self,
        bricks_prompt: List[List[str]],
        texts: List[str] = None,
        max_gen_len: int = 511,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        gt_demo: bool = True,
    ) -> Tuple[List[List[int]], List[List[str]],Optional[List[List[float]]]]:

        new_data = relative_transfer(bricks_prompt, self.brick_tokenizer)
        relative_models=new_data['model']
        relative_chooses=new_data['choose']

        models_id, chooses_id = self.brick_tokenizer.tokenize(relative_models, relative_chooses, if_eos=False)

        # print(models_id[0])
        # print(models_id[1])

        params = self.model.params
        bsz = len(models_id)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in models_id)
        max_prompt_len = max(len(t) for t in models_id)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        
        # models_id的pad调整
        pad_id=self.brick_tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(models_id):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        # chooses的pad调整
        chooses = torch.full((bsz, total_len), pad_id, dtype=torch.long,device='cuda')
        for k, t in enumerate(chooses_id):
            chooses[k, : len(t)] = torch.tensor(t, dtype=torch.long,device='cuda')

        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_brick_mask = tokens != pad_id

        if not gt_demo:
            for cur_pos in range(min_prompt_len, total_len):
                token_logits, choose_logits = self.model.forward(
                                        tokens=tokens[:, prev_pos:cur_pos],
                                        chooses=chooses[:, prev_pos:cur_pos],
                                        texts=texts,
                                        # pad_mask=pad_mask,
                                        start_pos=prev_pos, 
                                        autoregressive=True
                                        )

                next_token=sample_logits(temperature,token_logits,top_p).reshape(-1)
                next_choose=sample_logits(temperature,choose_logits,top_p).reshape(-1)
               
                # only replace token if prompt has already been generated
                next_token = torch.where(input_brick_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                next_choose = torch.where(input_brick_mask[:, cur_pos], chooses[:, cur_pos], next_choose)
                
                tokens[:, cur_pos] = next_token
                chooses[:, cur_pos] = next_choose
                
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                        input=token_logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                
                # eos token和choose都可以，不知道怎样效果会好
                eos_reached |= (~input_brick_mask[:, cur_pos]) & (
                    (next_token == self.brick_tokenizer.eos_id) | (next_choose == self.brick_tokenizer.eos_id)
                )
                prev_pos = cur_pos
                if all(eos_reached):
                    break

            if logprobs:
                token_logprobs = token_logprobs.tolist()
            out_tokens, out_chooses, in_tokens, in_chooses = [],[], [],[]
            for i, toks in enumerate(tokens.tolist()):
                start = 0 if echo else len(models_id[i])
                in_toks = toks[:start]
                toks = toks[start : len(models_id[i]) + max_gen_len]
                chos = chooses[i, start : len(models_id[i]) + max_gen_len].tolist()
                
                if self.brick_tokenizer.eos_id in toks:
                    token_eos_idx = toks.index(self.brick_tokenizer.eos_id)
                    if self.brick_tokenizer.eos_id in chos:
                        chos_eos_idx = chos.index(self.brick_tokenizer.eos_id)
                        eos_idx = min(token_eos_idx, chos_eos_idx)
                        # eos_idx = token_eos_idx
                    else:
                        eos_idx = token_eos_idx
                elif self.brick_tokenizer.eos_id in chos:
                    eos_idx = chos.index(self.brick_tokenizer.eos_id)
                else:
                    eos_idx=len(toks)
                
                toks = toks[:eos_idx]
                chos = chos[:eos_idx]

                out_tokens.append(toks)
                out_chooses.append(chos)
                in_tokens.append(in_toks)

        else:
            token_logits, choose_logits = self.model.forward(
                                    tokens=tokens,
                                    chooses=chooses,
                                    texts=texts,
                                    # pad_mask=pad_mask,
                                    start_pos=0
                                    )
            predicted_tokens = sample_logits(temperature,token_logits,top_p,all=True)
            predicted_chooses = sample_logits(temperature,choose_logits,top_p,all=True).tolist()

            # output
            out_tokens, out_chooses, in_tokens, in_chooses = [],[],[],[]

            for i, toks in enumerate(predicted_tokens.tolist()):
                in_toks = tokens[i,1:len(models_id[i])].tolist()
                in_chos = chooses[i,1:len(models_id[i])].tolist()

                toks = toks[:len(models_id[i])-1]
                chos = predicted_chooses[i][:len(models_id[i])-1]

                if self.brick_tokenizer.eos_id in toks:
                    token_eos_idx = toks.index(self.brick_tokenizer.eos_id)
                    if self.brick_tokenizer.eos_id in chos:
                        chos_eos_idx = chos.index(self.brick_tokenizer.eos_id)
                        eos_idx = min(token_eos_idx, chos_eos_idx)
                    else:
                        eos_idx = token_eos_idx
                elif self.brick_tokenizer.eos_id in chos:
                    eos_idx = chos.index(self.brick_tokenizer.eos_id)
                else:
                    eos_idx=len(toks)
                
                toks = toks[:eos_idx]
                chos = chos[:eos_idx]

                out_tokens.append(toks)
                out_chooses.append(chos)
                in_tokens.append(in_toks)
                in_chooses.append(in_chos)
              
        # out_tokens:batch_size*generate_seq_len
 
        return out_tokens, out_chooses, in_tokens, in_chooses

    def lego_generation(
        self,
        prompts: List[List[str]],
        texts: List[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        gt_demo: bool =True,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
    
        self.model.eval()
        generation_tokens, generation_chooses, in_tokens,in_chooses = self.generate(
            bricks_prompt=prompts,
            texts=texts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            gt_demo=gt_demo
        )
        
        # generation_models_info:batch_size*seq_len
        generation_models_info=self.brick_tokenizer.detokenize(generation_tokens, generation_chooses, prompts, gt_demo,in_tokens,in_chooses)
        
        # if logprobs:
        #     return [
        #         {
        #             "generation": model_info,
        #             "tokens": [brick_info for brick_info in model_info],
        #             "logprobs": logprobs_i,
        #         }
        #         for model_info, logprobs_i in zip(generation_models_info, generation_logprobs)
        #     ]
        return [{"generation": model_info} for model_info in generation_models_info]
