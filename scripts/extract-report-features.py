import argparse
import os

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from tqdm import trange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to report CSV.",
    )
    parser.add_argument(
        "--output_h5ad",
        required=True,
        help="Path to save extracted report features.",
    )
    parser.add_argument(
        "--feature_label",
        default="text",
        help="How to label the feature within the dataframe"
    )
    parser.add_argument(
        "--repo_root",
        help="Root directory for running",
    )
    parser.add_argument(
        "--model",
        choices=["biomistral", "conch", "llama"],
        default="biomistral",
        help="Foundation model to use to extract report features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference. Only used if model does not use vLLM",
    )
    args = parser.parse_args()

    return args


def biomistral_extract(repo_root: str, reports: list) -> np.ndarray:
    from transformers import MistralModel
    from vllm import LLM

    biomistral_path = os.path.join(repo_root, "models", "BioMistral-7B")
    if not os.path.exists(biomistral_path):  # not found locally
        biomistral_path = "BioMistral/BioMistral-7B"  # fall back to HF

    biomistral_base = os.path.join(repo_root, "models", "BioMistral-7B-base")
    if not os.path.exists(biomistral_base):  # need to sanitize state dict
        # BioMistral on HF is configured as MistralForCausalLM
        # and the transformer weights are prefixed with "model.".
        # vLLM requires the model be configured as MistralModel for embeddings
        # so load using huggingface (which takes care of weight prefixes too)
        # and save just the transformer backbone model.
        temp = MistralModel.from_pretrained(biomistral_path, torch_dtype=torch.bfloat16)
        temp.save_pretrained(biomistral_base, safe_serialization=False)  # TODO errors?
        del temp

    model = LLM(model=biomistral_base, tokenizer=biomistral_path, enforce_eager=True)
    outputs = model.encode(reports)
    embeds = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
    return embeds

def llama_extract(repo_root: str, reports: list) -> np.ndarray:
    from transformers import LlamaModel
    from vllm import LLM

    llama_path = "meta-llama/Llama-3.1-8B-Instruct"

    # biomistral_base = os.path.join(repo_root, "models", "BioMistral-7B-base")
    # if not os.path.exists(biomistral_base):  # need to sanitize state dict
    #     # BioMistral on HF is configured as MistralForCausalLM
    #     # and the transformer weights are prefixed with "model.".
    #     # vLLM requires the model be configured as MistralModel for embeddings
    #     # so load using huggingface (which takes care of weight prefixes too)
    #     # and save just the transformer backbone model.
    #     temp = MistralModel.from_pretrained(llama_path, torch_dtype=torch.bfloat16)
    #     temp.save_pretrained(biomistral_base, safe_serialization=False)  # TODO errors?
    #     del temp

    model = LLM(model=llama_path, tokenizer=llama_path, enforce_eager=True)
    outputs = model.encode(reports)
    embeds = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
    return embeds


def conch_extract(repo_root: str, reports: list, batch_size: int) -> np.ndarray:
    from conch.open_clip_custom import (
        create_model_from_pretrained,
        get_tokenizer,
        tokenize,
    )

    tokenizer = get_tokenizer()

    model_path = os.path.join(repo_root, "models", "CONCH", "pytorch_model.bin")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", model_path)
    model.eval()
    model.to("cuda")

    embeds = []
    for i in trange(0, len(reports), batch_size):
        batch_reports = reports[i : i + batch_size]
        with torch.inference_mode():
            tokens = tokenize(texts=batch_reports, tokenizer=tokenizer).to("cuda")
            batch_embeds = model.encode_text(tokens, normalize=True, embed_cls=True)
            batch_embeds = batch_embeds.detach().cpu().numpy()
            embeds.append(batch_embeds)
    embeds = np.concatenate(embeds)
    return embeds


if __name__ == "__main__":
    args = parse_args()
    assert not os.path.exists(args.output_h5ad)
    if not args.repo_root:
        args.repo_root = os.path.dirname(os.path.dirname(__file__))

    report_df = pd.read_csv(args.input_csv)
    assert "case_id" in report_df and "text" in report_df

    if args.model == "biomistral":
        embeds = biomistral_extract(
            repo_root=args.repo_root,
            reports=report_df["text"].to_list(),
        )
    elif args.model == "conch":
        embeds = conch_extract(
            repo_root=args.repo_root,
            reports=report_df["text"].to_list(),
            batch_size=args.batch_size,
        )
    elif args.model == "llama":
        embeds = llama_extract(
            repo_root=args.repo_root,
            reports=report_df["text"].to_list(),
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    out_adata = AnnData(obs=pd.DataFrame(index=report_df["case_id"].to_list()))
    out_adata.obsm[f"X_{args.feature_label}"] = embeds
    out_adata.uns["model_text"] = args.model
    out_adata.write_h5ad(args.output_h5ad)
