from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import gc
import os

model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

batch_size = 256

llm = LLM(
    model=model_id,
    tensor_parallel_size=2,
    enforce_eager=True,
    max_num_seqs=256,
    gpu_memory_utilization=0.96,
    max_model_len=1024,
    max_seq_len_to_capture=512,
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.15,
    max_tokens=256,
    stop_token_ids=[],
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Set padding token
if tokenizer.pad_token is None:
    print("Yes")
    tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id

def process_batch(entity_names, ocr_texts):
    prompts = [
        f'''Extract the numerical value and unit for "{entity_name}" from the text below. Strictly return the result in JSON format as: "{{'{entity_name}': 'value unit'}}".\nText: "{ocr_text}"'''
        for entity_name, ocr_text in zip(entity_names, ocr_texts)
    ]
    
    system_message = "You are a system assistant designed to format text and return the dimensions as asked."
    
    # Prepare all inputs in a single batch
    inputs = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ],
        tokenize=False,
        add_generation_prompt=True,
        padding=True,
    )
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = [output.outputs[0].text for output in outputs]
    
    del prompts
    gc.collect()
    
    return results


def extract_measurement(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            json_data = json.loads(json_str)
            for key, value in json_data.items():
                match = re.match(r'(\d+(?:\.\d+)?)\s*(\w+)', value)
                if match:
                    return f"{match.group(1)} {match.group(2)}"
        except json.JSONDecodeError:
            pass
    return None

# Assume we have these lists from the CSV file
df = pd.read_csv('extracted/cumulative_extracted_data.csv')
df = df[['ocr_text', 'entity_name', 'index', 'link']]
data_dict = df.to_dict(orient='records')


output_dir = 'llama_output/2'
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(0, len(data_dict), batch_size), desc="Processing Batches"):
    batch = data_dict[i:i+batch_size]
    
    batch_entity_names = [item["entity_name"] for item in batch]
    batch_ocr_texts = [item["ocr_text"] for item in batch]
    
    batch_file_path = os.path.join(output_dir, f"batch_{i}.json")
    
    results = process_batch(batch_entity_names, batch_ocr_texts)
    
    for item, result in zip(batch, results):
        item["measurement_result"] = result
    
    df_batch = pd.DataFrame(batch)
    
    df_batch.to_json(batch_file_path, orient='records', indent=2)
    
    print(f"Batch {i} saved to {batch_file_path}")

print("All batches processed and saved.")
