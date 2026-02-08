from transformers import AutoTokenizer

def prompt_len(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    msgs = [{"role":"user","content":"Hello"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tok(prompt).input_ids), prompt

for p in ["./KD_student_model_v3", "./awq_base"]:
    l, pr = prompt_len(p)
    print(p, "len:", l)
    print(pr[:200], "\n")
