# Demo Chat App

## Требования

### Для запуска нужно

* Зарегистрироваться в `Hugging Face` - <https://huggingface.co/>.
* Перейти на страницу модели - <https://huggingface.co/google/gemma-2b-it>.
* Согласиться с требованиями по использованию модели компании `Google`.
* В личном кабинете `Hugging Face`, вкладка `Access Tokens`, создать токен.
* В корне проекта создать `.streamlit\secrets.toml` со следующим содержимым `HUGGING_FACE_ACCESS_TOKEN = "Ваш токен"`

## Установка

Рекомендуется создать виртульную среду `venv` или `conda`.

### Установка зависимостей через Conda

    conda install conda-forge::transformers
    conda install conda-forge::streamlit
    conda install conda-forge::accelerate

### Установка зависимостей через PyPI

    pip install transformers
    pip install streamlit
    pip install accelerate

### Установка PyTorch под Windows

    CUDA: conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118
    CPU: conda install pytorch cpuonly -c pytorch
    CPU: pip install torch

## Запуск

    streamlit run app.py
