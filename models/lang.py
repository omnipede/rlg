import torch
import torch.nn.functional as F
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

    def generate(self, embeddings, max_length=20, temperature=1.) -> str:

        generated_list = []
        top_p = 0.9
        filter_value = -float("Inf")

        with torch.no_grad():

            entry_finished = False
            generated = embeddings
            generated_tokens = torch.tensor([]).unsqueeze(0)

            for i in range(max_length):
                outputs = self.model(inputs_embeds=generated)
                logits = outputs.logits.unsqueeze(0)

                logits = logits[:, -1, :] / temperature if temperature > 0 else 1.0

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                next_embedding = self.model.get_input_embeddings()(next_token).squeeze(dim=0)
                generated = torch.cat((generated, next_embedding), dim=0)
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

                if next_token in self.tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    output_list = list(generated_tokens.squeeze().numpy())
                    output_text = self.tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated_tokens.squeeze().numpy())
                output_text = f"{self.tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)

        return generated_list[0]
