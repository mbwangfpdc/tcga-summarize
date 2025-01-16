import csv

# Load reports from a file
def reports_from_file(filename) -> list[tuple[str, str]]:
    data = []
    with open(filename, newline="") as reports:
        reader = csv.DictReader(reports)
        for row in reader:
            data.append((row["case_id"], row["text"]))
    return data

# Dump reports into a file
def reports_to_file(filename, data: list[tuple[str, str]]):
    with open(filename, "w+", newline="") as reports:
        writer = csv.DictWriter(reports, fieldnames=["case_id", "text"])
        writer.writeheader()
        for row in data:
            writer.writerow({"case_id": row[0], "text": row[1]})

# Write two a pre and post transformation report into different files for offline comparison
def write_for_compare(input_str: str, transformed: str):
    with open("output/raw_text.txt", "a+") as base_out:
        base_out.write(input_str + "\n")
        base_out.write("======END OF PROMPT======\n")
    with open("output/summarized.txt", "a+") as test_out:
        test_out.write(transformed + "\n")
        test_out.write("======END OF OUTPUT======\n")

def reset_compare():
    try:
        import os
        os.remove("output/raw_text.txt", )
        os.remove("output/summarized.txt")
    except OSError:
        return
