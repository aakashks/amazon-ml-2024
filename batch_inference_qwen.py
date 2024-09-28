from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import pandas as pd
import json
from tqdm import tqdm
import os
import gc

image_folder = "~/data/images_test"
output_folder = "output/"
df_test = pd.read_csv("student_resource 3/dataset/test.csv")

BATCH_SIZE = 128
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

default_json = {
    "item_weight": "",
    "item_volume": "",
    "voltage": "",
    "wattage": "",
    "maximum_weight_recommendation": "",
    "height": "",
    "depth": "",
    "width": ""
}


cumulative_results = []


llm = LLM(
    model=MODEL_ID,
    dtype="half",
    enforce_eager=True,
    tensor_parallel_size=2,
    max_num_seqs=128,
    gpu_memory_utilization=0.96,
    max_model_len=3200,
    max_seq_len_to_capture=1024,
    limit_mm_per_prompt={"image": 1, "video": 0},
)


sampling_params = SamplingParams(
    temperature=0.4,
    top_p=0.1,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

def process_batch(batch):
    prompts = []
    
    for i in batch:
        entity_name = df_test.iloc[i].entity_name
        image_filename = os.path.basename(df_test.iloc[i].image_link)
        
        image_path = os.path.join(image_folder, image_filename)
        
        prompt_t = f"""
            Extract the following details in JSON format: 
            Focus on {entity_name}. Format the output as a JSON object: {{ "{entity_name}": <value with unit>}}
            """

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://" + image_path,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": prompt_t},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        prompts.append({'prompt': prompt, 'multi_modal_data': {'image': image_inputs}})
        

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    try:
        start_index = list(batch)[0]
        assert len(outputs) == len(batch)
        results = [
            {
                'index': int(df_test.at[i, 'index']),
                'entity_name': df_test.at[i, 'entity_name'],
                'vlm_output': out.outputs[0].text
            }
            for i, out in zip(batch, outputs)
        ]
    except AssertionError:
        results = [
            {
                'index': int(df_test.at[i + start_index, 'index']),
                'entity_name': df_test.at[i + start_index, 'entity_name'],
                'vlm_output': out.outputs[0].text
            }
            for i, out in enumerate(outputs)
        ]
    
    del prompts
    
    return results


def main(df_test, batch_size):
    ctr = 1
    for i in tqdm(range(0, len(df_test), batch_size), desc="Processing batches"):    
        batch_results = process_batch(range(i, i+batch_size))
        gc.collect()
        with open(os.path.join(output_folder, f'batch_output_{ctr}.json'), 'w') as outfile:
            json.dump(batch_results, outfile, indent=4)
            
        print(f"-----Batch {ctr} saved.-----")
        
        ctr += 1



if __name__ == "__main__":
    main(df_test, batch_size=BATCH_SIZE)
    