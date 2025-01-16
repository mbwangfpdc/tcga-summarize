import random

from reports import *

PROMPT = "Please repeat the diagnosis, clinical history, and clinician comments from this report without rewording it or adding additional text:"
# PROMPT = "Please remove the 'gross description' from this report, if it is present:"
PROMPT = "Extract and repeat the results of the following clinical report in a single paragraph. Focus on test results, diagnoses and clinical history. Omit the gross description. Do not acknowledge this prompt."
ACK = "Here is the report summary:"

def make_prompt(report: str) -> str:
    return f"{PROMPT}\nINPUT:\n{report}\nINPUT_DONE\nOUTPUT:\n"
def make_example(example_report: str, fixed_report: str) -> str:
    return f"{make_prompt(example_report)}{fixed_report}\nOUTPUT_DONE\n===\n"
def make_prompt_for_chat(case: str, report: str) -> str:
    return f"\n{PROMPT}\n{case} - {report}\n"

examples = reports_from_file("examples.csv")
examples_fixed = reports_from_file("examples_fixed.csv")
assert(len(examples) == len(examples_fixed))
report_hints = [(case, e, ef) for (case, e), (_, ef) in zip(examples, examples_fixed)]

TRAIN_NUM = 3
random.shuffle(report_hints)
report_hints = report_hints[:TRAIN_NUM]

# Make the full prompt including the preprompt
# def make_full_prompt(report: str) -> str:
#     return preprompt + make_prompt(report)
# TYPOS = "fix typos"
# DIAGNOSTIC = "exclude sections like method notes and gross description"
# CONCISE = "make sentences more concise"
# preprompt = f"Summarize this pathological report, but first {', '.join([DIAGNOSTIC])}: \n\n"
# preprompt = "Extract diagnostic sections from this report without rewording it or adding additional text:\n"
chat_preprompt = [
    {
        "role": "system",
        "content": "Please clean up the following text medical reports. Please do not modify or add to the reports unless instructed otherwise."
    },
]
for case, ex_in, ex_out in report_hints:
    chat_preprompt.append({"role": "user", "content": make_prompt_for_chat(case, ex_in)})
    chat_preprompt.append({"role": "assistant", "content": f"{ACK}\n{case} - {ex_out}"})
