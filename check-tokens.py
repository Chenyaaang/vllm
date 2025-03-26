from transformers import AutoTokenizer


model = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

# invalid tokens:
tokenizer.decode(2198)  # ","
tokenizer.decode(68882)     # "],"


# missing tokens
# 4913: '{"'