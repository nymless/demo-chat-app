import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "google/gemma-2b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)


def load_model(token):
    """Загрузка модели

    Returns:
        generate(messages): функция для генерации ответа
    """
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )

    def generate(messages):
        """Функция для генерации ответа

        Args:
            messages: список словарей сообщений чата

        Returns:
            str: ответ модели
        """
        # Реализованное в Hugging Face средство работы с чатом
        token_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Настроим модель в режим работы multinomial sampling
        # Информация: https://huggingface.co/docs/transformers/generation_strategies#decoding-strategies
        token_outputs = model.generate(
            input_ids=token_inputs,
            max_new_tokens=512,
            do_sample=True,
            num_beams=1,
            temperature=0.5,
        )
        # Т.к. ответ содержит всю историю чата, возьмём только последний ответ модели
        response_tokens = token_outputs[0][token_inputs.shape[-1] :]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response

    return generate
