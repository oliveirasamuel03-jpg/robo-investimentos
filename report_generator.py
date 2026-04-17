import json

def generate_report(metrics):
    with open("institutional_report.json", "w") as f:
        json.dump(metrics, f, indent=4)