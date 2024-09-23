from tensorrt_llm import LLM, SamplingParams

prompts = [
    "La capital de Francia es"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

modelo = "D:\DESARROLLO_EM\SX_AutoResumen\modelos\metallama31puro"

#llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm = LLM(model=modelo)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")