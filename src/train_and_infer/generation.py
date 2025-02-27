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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

# from model.model import ModelArgs, Transformer
from ..model.llama_model import ModelArgs, Transformer
from ..model.tokenizer import brick_tokenizer
from ..utils.utils import rotation_6d_to_9d, sample_top_p, seq_sample, images_transform

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        image_model_dir: str,
        image_model_size: str,
        original_bricks_path:str,
        new_bricks_path:str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        position_dim: int = 128,
        out_pad_dim: int = 1,
        rank: int = 2,
        c_n_heads: int = 32,
        patch_h: int = 32,
        patch_w: int =24,
        image_dim: int = 384,
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
                patch_h=patch_h,
                patch_w=patch_w,
                image_dim=image_dim,
                add_cross=add_cross,
                **params,
            )
        brick_tokenizer_instance=brick_tokenizer(new_bricks_path,original_bricks_path)
        model_args.vocab_size = brick_tokenizer_instance.n_words
        print(f'The vocab of LEGO is {model_args.vocab_size}')

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_dtype(torch.half)
        # torch.set_default_device('cuda')

        if not pretrain:
            image_model = torch.hub.load(image_model_dir, image_model_size, source='local').cuda()
            model = Transformer(model_args, image_model)
        else:
            model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model,new_bricks_path,original_bricks_path), checkpoint

    def __init__(self, model: Transformer,new_bricks_path,original_bricks_path):
        self.model = model
        self.brick_tokenizer=brick_tokenizer(new_bricks_path,original_bricks_path)
        self.rot_choose=torch.tensor(
                    [
                        [1,0,0, 0,1,0, 0,0,1],
                        [1,0,0, 0,0,1, 0,-1,0],
                        [1,0,0, 0,-1,0, 0,0,-1],
                        [1,0,0, 0,0,-1, 0,1,0],

                        [0,1,0, -1,0,0, 0,0,1],
                        [0,1,0, 0,0,-1, -1,0,0],
                        [0,1,0, 1,0,0, 0,0,-1],
                        [0,1,0, 0,0,1, 1,0,0],

                        [-1,0,0, 0,-1,0, 0,0,1],
                        [-1,0,0, 0,0,1, 0,1,0],
                        [-1,0,0, 0,1,0, 0,0,-1],
                        [-1,0,0, 0,0,-1, 0,-1,0],

                        [0,-1,0, 1,0,0, 0,0,1],
                        [0,-1,0, 0,0,1, -1,0,0],
                        [0,-1,0, -1,0,0, 0,0,-1],
                        [0,-1,0, 0,0,-1, 1,0,0],

                        [0,0,1, 1,0,0, 0,1,0],
                        [0,0,1, 0,1,0, -1,0,0],
                        [0,0,1, -1,0,0, 0,-1,0],
                        [0,0,1, 0,-1,0, 1,0,0],

                        [0,0,-1, 1,0,0, 0,-1,0],
                        [0,0,-1, 0,1,0, 1,0,0],
                        [0,0,-1, -1,0,0, 0,1,0],
                        [0,0,-1, 0,-1,0, -1,0,0],
                    ], dtype=torch.float32
                )
    
    @torch.inference_mode()
    def generate(
        self,
        bricks_prompt: List[List[str]],
        images: torch.Tensor = None,
        max_gen_len: int = 511,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        gt_demo: bool = True,
    ) -> Tuple[List[List[int]], List[List[str]],Optional[List[List[float]]]]:
        
        models_id,models_position=self.brick_tokenizer.devide_brick_and_postion(bricks_prompt,if_demo=True)
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
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_brick_mask = tokens != pad_id
        input_position_mask=input_brick_mask.unsqueeze(2)
        input_position_mask=input_position_mask.repeat(1,1,12)

        # models_position的pad调整
        pad_position=self.brick_tokenizer.pad_position[0]
        tokens_position=torch.full((bsz, total_len, 12), pad_position, dtype=torch.float16, device="cuda")
        for k, t in enumerate(models_position):
            tokens_position[k, : len(t),:] = torch.tensor(t, dtype=torch.float16, device="cuda")
        
        if not gt_demo:
            for cur_pos in range(min_prompt_len, total_len):
                logits, position = self.model.forward(
                                        tokens=tokens[:, prev_pos:cur_pos],
                                        tokens_position=tokens_position[:,prev_pos:cur_pos,:], 
                                        images=images, 
                                        start_pos=prev_pos, 
                                        autoregressive=True
                                        )
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                next_token_position=position[:,-1].reshape(bsz,9)

                trans=next_token_position[:,:3]
                rot_6d=next_token_position[:,3:]
                rot=rotation_6d_to_9d(rot_6d).view(-1,9)
                mse = torch.mean((rot.unsqueeze(1) - self.rot_choose.unsqueeze(0)) ** 2, dim=2)
                min_mse_indices = torch.argmin(mse, dim=1)
                rot_discrete=self.rot_choose[min_mse_indices]

                next_position=torch.cat((trans,rot_discrete),dim=1)
                

                # only replace token if prompt has already been generated
                next_token = torch.where(input_brick_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                next_position=torch.where(input_position_mask[:, cur_pos], tokens_position[:, cur_pos,:], next_position)
                
                tokens[:, cur_pos] = next_token
                tokens_position[:,cur_pos,:]=next_position
                
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                eos_reached |= (~input_brick_mask[:, cur_pos]) & (
                    next_token == self.brick_tokenizer.eos_id
                )
                prev_pos = cur_pos
                if all(eos_reached):
                    break

            if logprobs:
                token_logprobs = token_logprobs.tolist()
            out_tokens, out_tokens_position,out_logprobs = [],[], []
            for i, toks in enumerate(tokens.tolist()):
                start = 0 if echo else len(models_id[i])
                toks = toks[start : len(models_id[i]) + max_gen_len]
                toks_position=tokens_position[i,start:len(models_id[i]) + max_gen_len]
                
                probs = None
                if logprobs:
                    probs = token_logprobs[i][start : len(models_id[i]) + max_gen_len]
                # cut to eos tok if any
                if self.brick_tokenizer.eos_id in toks:
                    eos_idx = toks.index(self.brick_tokenizer.eos_id)
                    toks = toks[:eos_idx]
                    toks_position=toks_position[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                out_tokens.append(toks)
                out_tokens_position.append(toks_position)
                out_logprobs.append(probs)
        else:
            logits, position = self.model.forward(
                                    tokens=tokens, 
                                    tokens_position=tokens_position, 
                                    images=images,
                                    start_pos=0
                                    )
            # brick id
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                predicted_tokens = seq_sample(probs, top_p)
            else:
                predicted_tokens = torch.argmax(logits, dim=-1)

            # position            
            predicted_token_positions=position.reshape(-1,9)

            trans=predicted_token_positions[:,:3]
            trans=trans+tokens_position[:,:,:3].reshape(-1,3)

            rot_6d=predicted_token_positions[:,3:]
            rot=rotation_6d_to_9d(rot_6d).view(-1,9)
            mse = torch.mean((rot.unsqueeze(1) - self.rot_choose.unsqueeze(0)) ** 2, dim=2)
            min_mse_indices = torch.argmin(mse, dim=1)
            rot_discrete=self.rot_choose[min_mse_indices]

            predicted_position=torch.cat((trans,rot_discrete),dim=1).reshape(bsz,-1,12)

            # output
            out_tokens, out_tokens_position,out_logprobs = [],[], []

            for i, toks in enumerate(predicted_tokens.tolist()):
                toks = toks[:len(models_id[i])-1]
                toks_position=predicted_position[i,0:len(models_id[i])-1]
                
                try:
                    index = toks.index(next(tok for tok in toks if tok != self.brick_tokenizer.eos_id))
                    toks=toks[index:]
                except:
                    toks=[self.brick_tokenizer.BRICK_TOKEN['3020.dat'],2]

                # cut to eos tok if any
                if self.brick_tokenizer.eos_id in toks:
                    eos_idx = toks.index(self.brick_tokenizer.eos_id)
                    toks = toks[:eos_idx]
                    toks_position=toks_position[:eos_idx]

                out_tokens.append(toks)
                out_tokens_position.append(toks_position)
        
        # out_tokens:batch_size*generate_seq_len
        # out_tokens_postion:batch_size*generate_seq_len*12
        return (out_tokens, out_tokens_position ,out_logprobs if logprobs else None)

    def lego_generation(
        self,
        opt,
        prompts: List[List[str]],
        images: torch.Tensor = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        gt_demo: bool =True,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        if opt.pretrain:
            transformed_images=None
        else:
            images_transformed=[]
            for image in images:
                images_transformed.append(images_transform(image,opt))
            
            transformed_images=torch.stack(images_transformed).cuda()

        self.model.eval()
        generation_tokens, generation_tokens_position, generation_logprobs = self.generate(
            bricks_prompt=prompts,
            images=transformed_images,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            gt_demo=gt_demo
        )
        
        self.brick_tokenizer.get_colors(prompts, generation_tokens, gt_demo)
        
        # generation_models_info:batch_size*seq_len
        generation_models_info=self.brick_tokenizer.merge_brick_and_position(generation_tokens, generation_tokens_position)
        
        if logprobs:
            return [
                {
                    "generation": model_info,
                    "tokens": [brick_info for brick_info in model_info],
                    "logprobs": logprobs_i,
                }
                for model_info, logprobs_i in zip(generation_models_info, generation_logprobs)
            ]
        return [{"generation": model_info} for model_info in generation_models_info]
