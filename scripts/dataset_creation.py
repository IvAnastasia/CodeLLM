import json
import re
import subprocess
from pathlib import Path
import ast
from pathlib import Path

subprocess.run(["git", "clone", "https://github.com/scikit-learn/scikit-learn.git"])

repo_path = Path("scikit-learn")

def extract_functions_with_docstrings(code_path):
    examples = []
    for file in Path(code_path).rglob("*.py"):
        with open(file, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            body_lines = [ast.get_source_segment(open(file).read(), node)]
                            examples.append({
                                "prompt": docstring,
                                "completion": "\n".join(body_lines)
                            })
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return examples


data = extract_functions_with_docstrings('scikit-learn')


def is_trivial_function(completion):
    # Skip if it's just 'pass', '...', or too short
    lines = completion.strip().splitlines()
    body = [line.strip() for line in lines if line.strip() and not line.strip().startswith("def")]
    if len(body) == 0:
        return True
    if all(line in ("pass", "...") for line in body):
        return True
    return False

def clean_example(example, max_tokens=500):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    completion = re.sub(r'\"\"\"(.*?)\"\"\"', '', completion, flags=re.DOTALL)
    completion = re.sub(r'#.*', '', completion, flags=re.DOTALL)

    # Remove trivial completions
    if is_trivial_function(completion):
        return None

    return {
        "prompt": prompt,
        "completion": completion
    }

cleaned_data = []
for ex in data:
    cleaned = clean_example(ex)
    if cleaned:
        cleaned_data.append(cleaned)

with open("cleaned_dataset.jsonl", "w") as f:
    for item in cleaned_data:
        json.dump(item, f)
        f.write("\n")

#print(f"Kept {len(cleaned_data)} / {len(data)} examples after cleaning.")
