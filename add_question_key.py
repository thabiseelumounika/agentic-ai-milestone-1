
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
client = Client()

dataset_name = "ds-granular-oleo-34"
examples = list(client.list_examples(dataset_name=dataset_name))

print(f"Adding 'question' key for {len(examples)} examples...")

for ex in examples:
    # Use the existing 'task' or 'input' content
    content = ex.inputs.get("task") or ex.inputs.get("input")
    
    new_inputs = ex.inputs.copy()
    new_inputs["question"] = content
    
    client.update_example(ex.id, inputs=new_inputs)
    print(f"Updated {ex.id} with 'question' key.")

print("Done! Your {question} variable in the Playground will now work.")
