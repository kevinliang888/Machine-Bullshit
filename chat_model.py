from typing import List, Dict, Any
import os
from google.genai import types


openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")


def llama_chat(chat, tokenizer, model, max_new_tokens=512, output_scores=False, processors=None, temperature=1.0):
    chat_tokens = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt').to("cuda")
    # chat_tokens = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt').to(model.device)
    outputs = model.generate(chat_tokens, logits_processor=processors, do_sample=True, max_new_tokens=max_new_tokens, 
                             return_dict_in_generate=True, output_scores=output_scores, pad_token_id=tokenizer.eos_token_id)
    new_chat_str = tokenizer.decode(outputs.sequences[0])

    return outputs, new_chat_str


def get_response(output, model_name="Llama-3"):
    if "Llama-3" in model_name or "llama3" in model_name:
        # additional check
        if "<|start_header_id|>" in output.split("<|end_header_id|>")[-1]:
            response = output.split("<|start_header_id|>")[-1].split("<|eot_id|>")[0].strip()
        else:
            # This is original 
            response = output.split("<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "Llama-2" in model_name or "llama2" in model_name:
        response = output.split('[/INST]')[-1].split("</s>")[0].strip()
    return response


class ChatLLM:
    def __init__(self, provider: str, model: str, **backend_kwargs):
        self.provider, self.model = provider.lower(), model
        self._init_backend(**backend_kwargs)

    def _init_backend(self, **kw):
        if self.provider == "openai":               # GPT‑4‑o3
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            openai.api_key = openai_api_key
            self.client = openai.OpenAI(api_key=openai_api_key)

        elif self.provider == "hf":                 # Llama‑3‑8B (HF / local)
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.tok  = AutoTokenizer.from_pretrained(self.model)
            # self.mdl  = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.float16)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tok,
                device_map="auto",
                torch_dtype="float16",
                model_kwargs={"low_cpu_mem_usage": True}
            )

        elif self.provider == "anthropic":          # Claude
            import anthropic
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not claude_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
            self.client = anthropic.Anthropic(api_key=claude_api_key)

        elif self.provider == "gemini":             # Gemini-Pro / Vision

            from google import genai
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise RuntimeError("GOOGLE_API_KEY not set in environment")

            self.client = genai.Client(api_key=gemini_api_key)

        else:
            raise ValueError("Unsupported provider")

    def chat(self, messages: List[Dict[str, str]], **gen_kw) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, **gen_kw
            )
            return resp.choices[0].message.content.strip()

        elif self.provider == "hf":
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out = self.pipe(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0
            )[0]["generated_text"]

            return out[len(prompt):].strip()

        elif self.provider == "anthropic":
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            convo = [m for m in messages if m["role"] in {"user", "assistant"}]
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_msg,
                messages=convo
            )
            print(message.content[0].text)
            print("---------")
            return message.content[0].text

        elif self.provider == "gemini":
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_content = [m['content'] for m in messages if m["role"]  == "user"][0]
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg),
                contents=user_content
            )
            return response.text