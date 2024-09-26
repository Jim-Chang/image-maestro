import io
from datetime import datetime

import requests
import streamlit as st
from PIL import Image, PngImagePlugin

from app.config import set_google_api_key, set_fal_api_key, set_anthropic_api_key
from app.image_generator import ImageSize, generate_image
from app.image_prompt_generator_chain import run_image_prompt_generator_chain
from app.models import FalAiModel, AnthropicModel, GoogleModel


def add_to_user_input_history(user_input: str):
    user_input_history = st.session_state.get("user_input_history", [])
    user_input_history.append(user_input)
    st.session_state["user_input_history"] = user_input_history


def on_submit():
    user_input = st.session_state.get("user_input")
    llm_mode = st.session_state.get("select_llm_model")
    image_model = st.session_state.get("select_image_model")
    image_size = st.session_state.get("select_image_size")
    num_images = st.session_state.get("num_images")
    num_inference_steps = st.session_state.get("num_inference_steps")
    guidance_scale = st.session_state.get("guidance_scale")

    add_to_user_input_history(user_input)
    st.session_state["user_input"] = ""

    print("run image prompt generator chain...")
    image_prompt = run_image_prompt_generator_chain(user_input, llm_mode)
    st.session_state["image_prompt"] = image_prompt
    print("======= image_prompt =======")
    print(image_prompt)

    print("generate image...")
    image_urls = generate_image(
        prompt=image_prompt.english_prompt,
        model=image_model,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
    )
    st.session_state["image_urls"] = image_urls
    print("======= image_urls =======")
    print(image_urls)


def on_regenerate():
    image_model = st.session_state.get("select_image_model")
    image_size = st.session_state.get("select_image_size")
    num_images = st.session_state.get("num_images")
    num_inference_steps = st.session_state.get("num_inference_steps")
    guidance_scale = st.session_state.get("guidance_scale")
    image_prompt = st.session_state.get("image_prompt")

    print("generate image...")
    image_urls = generate_image(
        prompt=image_prompt.english_prompt,
        model=image_model,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
    )
    st.session_state["image_urls"] = image_urls
    print("======= image_urls =======")
    print(image_urls)


def download_image_and_add_exif(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    if img.mode != "RGB":
        img = img.convert("RGB")

    parameters = {
        "user_input": st.session_state.get("user_input_history")[-1],
        "image_prompt": st.session_state.get("image_prompt"),
        "llm_model": st.session_state.get("select_llm_model"),
        "image_model": st.session_state.get("select_image_model"),
        "num_images": st.session_state.get("num_images"),
        "num_inference_steps": st.session_state.get("num_inference_steps"),
        "guidance_scale": st.session_state.get("guidance_scale"),
    }
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Parameters", str(parameters))

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG", pnginfo=metadata)
    img_buffer.seek(0)
    return img_buffer


def layout():
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.form("chat_form"):
            st.text_input("訊息", key="user_input")
            st.form_submit_button("發送", on_click=on_submit)

            st.selectbox(
                "文字模型",
                [
                    AnthropicModel.claude_3_5_sonnet_20240620,
                    GoogleModel.gemini_15_pro_exp_0827,
                ],
                index=0,
                key="select_llm_model",
            )

            st.selectbox(
                "生圖模型",
                [
                    FalAiModel.flux_dev,
                    FalAiModel.flux_schnell,
                    FalAiModel.flux_pro,
                    FalAiModel.flux_realism,
                ],
                index=0,
                key="select_image_model",
            )
            st.selectbox(
                "圖片大小",
                ImageSize,
                index=3,
                key="select_image_size",
            )
            st.slider("產生張數", min_value=1, max_value=5, value=1, key="num_images")
            st.slider(
                "迭代步數",
                min_value=30,
                max_value=50,
                value=40,
                key="num_inference_steps",
            )
            st.slider(
                "引導比例",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                key="guidance_scale",
            )

        user_input_history = st.session_state.get("user_input_history")
        if user_input_history:
            for user_input in user_input_history[::-1]:
                st.chat_message("human").write(user_input)

    with col_right:
        image_prompt = st.session_state.get("image_prompt")
        if image_prompt:
            st.subheader("Image Prompt", divider="rainbow")
            st.write(image_prompt.english_prompt)
            st.write(image_prompt.chinese_prompt)
            st.button("重新以此描述生成圖片", on_click=on_regenerate)

        image_urls = st.session_state.get("image_urls")
        if image_urls:
            st.subheader("Generated Image", divider="rainbow")

            for idx, image_url in enumerate(image_urls):
                img_buffer = download_image_and_add_exif(image_url)
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                file_name = f"{current_time}_{idx + 1}.png"

                st.download_button(
                    label=f"Download Image {idx + 1}",
                    data=img_buffer,
                    file_name=file_name,
                    mime="image/jpeg",
                )
                st.image(image_url)


def main():
    layout()


if __name__ == "__main__":
    set_google_api_key()
    set_anthropic_api_key()
    set_fal_api_key()

    st.set_page_config(layout="wide")
    main()
