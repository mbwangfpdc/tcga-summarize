import csv
# This script updates the headers of the reports. It doesn't modify content besides making the case number more obvious.
with open("TCGA_Reports.csv") as rin:
    reader = csv.DictReader(rin)
    with open("TCGA_Reports_Processed.csv", "w+") as rout:
        writer = csv.DictWriter(rout, fieldnames=["case", "report"])
        writer.writeheader()
        for r in reader:
            writer.writerow({"case": r["patient_filename"].split(".")[0], "report": r["text"]})
