import textwrap
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import GoogleModel


class _Prompt:
    system = textwrap.dedent(
        """
        You are a visual artist, skilled at creating images.
        - Please carefully review the user's needs and description, and design an image that fits their request.
        - Describe the image in detail.
        - Choose the style of the image—such as photography or painting—based on the user's context and needs, and specify the style in your description.
        - If photography is chosen, specify the camera model, lens, film if need, according to the user's requirements.
        - If the user provides specific settings, follow them exactly.
        - Provide only the image description, avoiding mentioning unrelated details.
        - Limit your response to no more than 20 sentences.
        - Use American English for your description.
        """
    )

    human = textwrap.dedent(
        """
        Please create an image based on the following requirements.
        Just provide the image description, avoid anything else.
        
        # User's Request:
        {user_input}
        """
    )


class _Model:
    @staticmethod
    def build():
        gemini_15_model = ChatGoogleGenerativeAI(
            model=GoogleModel.gemini_15_pro_exp_0827,
            temperature=0.8,
        )

        return gemini_15_model


class _PromptTemplate:
    @staticmethod
    def build():
        return ChatPromptTemplate.from_messages(
            [
                ("system", _Prompt.system),
                ("human", _Prompt.human),
            ]
        )


def _build_chain():
    model = _Model.build()
    prompt = _PromptTemplate.build()
    parser = StrOutputParser()

    _chain = (
            {
                "user_input": itemgetter("user_input"),
            }
            | prompt
            | model
            | parser
    )

    return _chain


def run_image_prompt_generator_chain(user_input: str) -> str:
    _chain = _build_chain()
    return _chain.invoke({"user_input": user_input})
