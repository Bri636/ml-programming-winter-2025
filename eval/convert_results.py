import json

def transform_jsonl(input_file: str, output_file: str):
    """
    Transforms each line from:
        {
          "task_id": "...",
          "completion": [{"generated_text": "some string ..."}]
        }
    into:
        {
          "task_id": "...",
          "completion": "some string ..."
        }
    and writes to a new JSONL file.
    """
    transformed_records = []

    # 1. Read each line as JSON
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            
            # 2. Extract "generated_text" from the 'completion' list (assuming it exists)
            if "completion" in data and isinstance(data["completion"], list) and data["completion"]:
                # Typically something like [{"generated_text": "..."}]
                gen_text = data["completion"][0].get("generated_text", "")
            else:
                gen_text = ""

            # 3. Replace the 'completion' field with just the string
            data["completion"] = gen_text

            transformed_records.append(data)

    # 4. Write the new structure to a JSONL output file
    with open(output_file, "w", encoding="utf-8") as fout:
        for record in transformed_records:
            fout.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # Usage:
    input_path = "/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-program-winter-2025/eval/eval_results/merged.jsonl"
    output_path = "/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-program-winter-2025/eval/eval_results/clean_merged.jsonl"

    transform_jsonl(input_path, output_path)
    print(f"Transformed JSONL saved to: {output_path}")
