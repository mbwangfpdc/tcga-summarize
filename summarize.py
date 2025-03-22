import os
import argparse
import random

from copy import deepcopy

from vllm import LLM, SamplingParams
from reports import *

random.seed(42)

DATA_DIR = "data"
OUTPUT_DIR = "output"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", choices=["bm", "llama"])
    parser.add_argument("--testonly", action="store_true", default=False)
    parser.add_argument("--reports")
    parser.add_argument("--out", default="TCGA_Reports_Transformed.csv")
    return parser.parse_args()

args = parse_args()

# Load reports
reports_path = args.reports
reports = reports_from_file(reports_path)
random.shuffle(reports)

model_map = {
    "bm": "BioMistral/BioMistral-7B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct"
}

args.model = model_map[args.model]

# Load model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model = args.model
llm = LLM(model=model, enforce_eager=True)

PROMPT = "Extract and repeat the results of the following clinical report in a single paragraph. Focus on test results, diagnoses and clinical history. Omit the gross description. Do not acknowledge this prompt."
def make_prompt_for_chat(case: str, report: str) -> str:
    return f"\n{PROMPT}\n{case} - {report}\n"

# Mistral models don't do well with the chat history
chat_preprompt = []

NUM = 20 if args.testonly else len(reports)
convos = [deepcopy(chat_preprompt) for _ in range(NUM)]
if args.testonly:
    # For tests, shuffle the reports so we can do a diversity of testing
    myreports = deepcopy(reports)
    random.shuffle(myreports)
else:
    myreports = reports
for convo, (case, report) in zip(convos, myreports):
    convo.append({"role": "user", "content": make_prompt_for_chat(case, report)})
sampling_params = SamplingParams(max_tokens=1000, temperature=0, repetition_penalty=1.1)
outputs = llm.chat(messages=convos, sampling_params=sampling_params)

if args.testonly:
    reset_compare()
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        write_for_compare(prompt, generated_text)
else:
    new_reports = []
    for output, (case, _) in zip(outputs, myreports):
        generated_text = output.outputs[0].text
        generated_text = generated_text.replace("\n", " ")
        new_reports.append((case, generated_text))
    reports_to_file(args.out, new_reports)
print("Done!")
