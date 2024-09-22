import streamlit as st

from app.config import set_google_api_key, set_fal_api_key
from app.image_generator import generate_image, ImageSize
from app.image_prompt_generator_chain import run_image_prompt_generator_chain
from app.models import FalAiModel


def add_to_user_input_history(user_input: str):
    user_input_history = st.session_state.get("user_input_history", [])
    user_input_history.append(user_input)
    st.session_state["user_input_history"] = user_input_history


def on_submit():
    user_input = st.session_state.get("user_input")
    add_to_user_input_history(user_input)
    st.session_state["user_input"] = ""

    print("run image prompt generator chain...")
    image_prompt = run_image_prompt_generator_chain(user_input)
    st.session_state["image_prompt"] = image_prompt
    print("======= image_prompt =======")
    print(image_prompt)

    print("generate image...")
    image_urls = generate_image(
        prompt=image_prompt,
        model=FalAiModel.flux_realism,
        image_size=ImageSize.landscape_4_3,
        num_inference_steps=40,
        guidance_scale=3.5,
        num_images=1,
    )
    st.session_state["image_urls"] = image_urls
    print("======= image_urls =======")
    print(image_urls)


def layout():
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.form("chat_form"):
            st.text_input("訊息", key="user_input")
            st.form_submit_button("發送", on_click=on_submit)

            st.selectbox(
                "生圖模型",
                FalAiModel,
                index=0,
                key="select_image_model",
            )
            st.selectbox(
                "圖片大小",
                ImageSize,
                index=3,
                key="select_image_size",
            )
            st.slider("產生張數", min_value=1, max_value=5, value=1)
            st.slider("迭代步數", min_value=30, max_value=50, value=40)
            st.slider("引導比例", min_value=1.0, max_value=5.0, value=3.5)

        user_input_history = st.session_state.get("user_input_history")
        if user_input_history:
            for user_input in user_input_history[::-1]:
                st.chat_message("human").write(user_input)

    with col_right:
        image_prompt = st.session_state.get("image_prompt")
        if image_prompt:
            st.subheader("Image Prompt", divider="rainbow")
            st.write(image_prompt)

        image_urls = st.session_state.get("image_urls")
        if image_urls:
            st.subheader("Generated Image", divider="rainbow")
            for image_url in image_urls:
                st.image(image_url)


def main():
    layout()


if __name__ == "__main__":
    set_google_api_key()
    set_fal_api_key()

    st.set_page_config(layout="wide")
    main()
