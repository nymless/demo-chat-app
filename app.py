import streamlit as st
from model import load_model

st.title("Assistant Chat")


# Загрузка и кеширование модели
@st.cache_resource()
def load_model_cached():
    return load_model(st.secrets["HUGGING_FACE_ACCESS_TOKEN"])


# Функция для генерации ответа
generate = load_model_cached()

# Инициализация истории чата в сессионном хранилище
if "messages" not in st.session_state:
    st.session_state.messages = []

# Показ сообщений из истории чата. Использование модуля Streamlit для чата.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Получение ввода пользователя
if input := st.chat_input("What is up?"):
    # Добавление сообщения пользователя в историю чата
    st.session_state.messages.append({"role": "user", "content": input})
    # Показ сообщения пользователя в окне
    with st.chat_message("user"):
        st.markdown(input)

    # Создадим глубокую копию истории чата
    messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    # Передача модели копии истории чата. Получение ответа.
    response = generate(messages)
    # Добавление ответа в историю чата
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Показ ответа в окне
    with st.chat_message("assistant"):
        st.markdown(response)
