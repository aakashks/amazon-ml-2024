## Prompts
QWEN 2 VL
```
ONLY ONE IMAGE IS PROVIDED TO YOU. Extract textual features notably height, Depth, Width, Maximum Weight Recommendation, Item Weight, Voltage, Wattage, Item_volume whichever visible STRICTLY.
```
LLAMA 3.1
```
f'''Extract the numerical value and unit for "{entity_name}" from the text below. Strictly return the result in JSON format as: "{{'{entity_name}': 'value unit'}}".\nText: "{ocr_text}"'''
for entity_name, ocr_text in zip(entity_names, ocr_texts)
```



## Model hosting (for Testing purposes)

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8080 --tensor-parallel-size 2 --dtype=half --max-model-len 3200 --trust-remote-code --limit-mm-per-prompt image=1 --enforce-eager
```
