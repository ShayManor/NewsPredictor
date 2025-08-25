import json


def reverse_dict(original: dict):
    reversed: dict = {}
    for key, value in original.items():
        if value not in reversed:
            reversed[value] = [key]
        else:
            reversed[value].append(key)
    return reversed


def process_results(path):
    with open(path, "r") as f:
        results: dict = json.load(f)
    rev: dict = reverse_dict(results)

    counts = {}
    for key in rev.keys():
        counts[key] = len(rev[key])
    print(counts)
    print(rev)

if __name__ == "__main__":
    process_results("results.json")
