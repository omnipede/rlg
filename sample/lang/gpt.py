import torch
from torch import nn
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import EncoderDecoderModel, BertTokenizer


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


def gpt_only_generation():
    # https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272
    # --> 좋은 예제

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    model: GPTNeoForCausalLM = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    model.eval()

    prompt = (
        " Is life worth living should I blast myself "
    )

    prompt2 = (
        " Is life worth living should I blast myself "
    )

    inputs_list = tokenizer([prompt, prompt2], return_tensors="pt", padding=True)

    input_ids = inputs_list.input_ids[0]
    input_ids2 = inputs_list.input_ids[1]
    outputs = model(input_ids=input_ids, labels=input_ids2)

    print(outputs.loss)


def encoder_decoder():
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "patrickvonplaten/bert2bert-cnn_dailymail-fp16", "patrickvonplaten/bert2bert-cnn_dailymail-fp16"
    )

    tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    x = model.generate(
        input_ids=inputs.input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    decoded = tokenizer.batch_decode(x)
    print(decoded)

    pass
