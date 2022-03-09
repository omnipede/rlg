import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


def gpt_neo_sample():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)


def gpt_neo_embedding_extraction():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    embedding_layer = model.transformer.wte

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    inputs_embeds = embedding_layer(input_ids).squeeze()
    print(inputs_embeds.size())

    generated = model(inputs_embeds=inputs_embeds)

    logits = generated.logits
    generated_tokens = torch.argmax(logits, 1)
    generated_text = tokenizer.batch_decode(generated_tokens)

    print(generated_text)
