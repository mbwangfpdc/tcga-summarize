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
        choices=["biomistral", "mistral", "conch", "llama"],
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


def mistral_extract(repo_root: str, reports: list) -> np.ndarray:
    return mistral_extract_impl(repo_root=repo_root, reports=reports, model_name="Mistral-7B-Instruct-v0.1", hd_path="mistralai/Mistral-7B-Instruct-v0.1")

def biomistral_extract(repo_root: str, reports: list) -> np.ndarray:
    return mistral_extract_impl(repo_root=repo_root, reports=reports, model_name="BioMistral-7B", hd_path="BioMistral/BioMistral-7B")

def mistral_extract_impl(repo_root: str, reports: list, model_name: str, hd_path: str) -> np.ndarray:
    from transformers import MistralModel
    from vllm import LLM

    mistral_path = os.path.join(repo_root, "models", model_name)
    if not os.path.exists(mistral_path):  # not found locally
        mistral_path = hd_path

    mistral_base = os.path.join(repo_root, "models", f"{model_name}-Base")
    if not os.path.exists(mistral_base):  # need to sanitize state dict
        # Mistral on HF is configured as MistralForCausalLM
        # and the transformer weights are prefixed with "model.".
        # vLLM requires the model be configured as MistralModel for embeddings
        # so load using huggingface (which takes care of weight prefixes too)
        # and save just the transformer backbone model.
        temp = MistralModel.from_pretrained(mistral_path, torch_dtype=torch.bfloat16)
        temp.save_pretrained(mistral_base, safe_serialization=False)  # TODO errors?
        del temp

    model = LLM(model=mistral_base, tokenizer=mistral_path, enforce_eager=True)
    outputs = model.encode(reports)
    embeds = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
    return embeds

def llama_extract(repo_root: str, reports: list) -> np.ndarray:
    raise NotImplementedError("No LlamaModel supported, only LlamaForCausalLM")


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
            print(tokens.shape)
            print(tokens.T[tokens.T.shape[0] - 5:].T[:10])
            exit(0)
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
    assert "case_id" in report_df
    text_cols = [col for col in report_df.columns if col.startswith("text")]
    assert len(text_cols) > 0
    for text_col in text_cols:
        report_slice = report_df[["case_id", text_col]]
        if args.model == "biomistral":
            embeds = biomistral_extract(
                repo_root=args.repo_root,
                reports=report_slice[text_col].to_list(),
            )
        elif args.model == "mistral":
            embeds = mistral_extract(
                repo_root=args.repo_root,
                reports=report_slice[text_col].to_list(),
            )
        elif args.model == "conch":
            embeds = conch_extract(
                repo_root=args.repo_root,
                reports=report_slice[text_col].to_list(),
                batch_size=args.batch_size,
            )
        elif args.model == "llama":
            embeds = llama_extract(
                repo_root=args.repo_root,
                reports=report_slice[text_col].to_list(),
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        out_adata = AnnData(obs=pd.DataFrame(index=report_slice["case_id"].to_list()))
        out_adata.obsm[f"X_{args.feature_label}"] = embeds
        out_adata.uns["model_text"] = args.model
        out_adata.write_h5ad(f"{args.output_h5ad}.{text_col}")
