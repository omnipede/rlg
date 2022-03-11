from transformers import GPTNeoForCausalLM, GPT2Tokenizer


class LM:

    def __init__(self):
        """
        가사를 입력 받는 언어 모델
        """
        pretrained_gpt_model = "EleutherAI/gpt-neo-125M"
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(pretrained_gpt_model)
        self.model: GPTNeoForCausalLM = GPTNeoForCausalLM.from_pretrained(pretrained_gpt_model)

    def encode(self, text: str):
        """
        텍스트를 token 으로 변환하는 메소드
        """
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        return self.model.get_input_embeddings()(input_ids).squeeze(dim=0)

    def decode(self, tokens):
        """
        Embedding 을 입력받아 변환하는 메소드
        """
        # TODO How to use .generate() method with inputs_embeds ?
        output = self.model.generate(inputs_embeds=tokens, max_length=100)
        return self.tokenizer.batch_decode(output)[0]
