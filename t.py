import lmstudio as lms

model = lms.llm("qwen/qwen3-4b-2507")
result = model.respond("Who is Shah Rukh Khan?")

print(result)