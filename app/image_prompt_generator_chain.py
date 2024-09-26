import textwrap
from operator import itemgetter

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.models import GoogleModel, AnthropicModel


class _Prompt:
    system = textwrap.dedent(
        """
        You're an artistic creator skilled in crafting visual scenes. 
        Carefully review the user's needs and descriptions, then design and create a scene that meets their requirements. 
        - Describe the scene in detail.
        - Choose an appropriate style for the scene, such as photography or painting, based on the user's context and needs.
        - Specify the style in your description. For photography, select a camera, digital or film format, and lens that fit the user's situation and requirements. 
        - If the user specifies any settings, use those as given. 
        - Provide only the scene description, avoiding unrelated topics. 
        - Keep your response to 30 sentences or fewer.
        - Use American English in your description, then provide a traditional Chinese translation of that description.
        
        <Example>
        User's Request: 美國經典肌肉車，美國西部，韓國性感車模
        Result: Vintage American muscle car, gleaming red 1969 Chevrolet Camaro SS, parked on a dusty road in the American West. Rugged desert landscape with towering red rock formations in the background. Golden hour sunlight casts long shadows. Korean female model in a silver sequined mini dress poses seductively on the car's hood, her long black hair flowing in the breeze. Captured with a Canon EOS R5 digital camera, 24-70mm f/2.8 lens. Cinematic color grading emphasizes warm tones and high contrast.
        </Example>        
        """
    )

    human = textwrap.dedent(
        """
        User's Request: {user_input}
        
        {format_instructions}
        """
    )


class _Model:
    @staticmethod
    def build():
        gemini_15_model = ChatGoogleGenerativeAI(
            model=GoogleModel.gemini_15_pro_exp_0827,
            temperature=0.8,
        )

        claude3_sonnet_model = ChatAnthropic(
            model=AnthropicModel.claude_3_5_sonnet_20240620,
            temperature=0.8,
            max_tokens=4000,
        )

        return claude3_sonnet_model.configurable_alternatives(
            ConfigurableField(id="model"),
            default_key=AnthropicModel.claude_3_5_sonnet_20240620,
            **{
                GoogleModel.gemini_15_pro_exp_0827: gemini_15_model,
            },
        )


class _PromptTemplate:
    @staticmethod
    def build():
        return ChatPromptTemplate.from_messages(
            [
                ("system", _Prompt.system),
                ("human", _Prompt.human),
            ]
        )


class ImagePrompt(BaseModel):
    english_prompt: str = Field(
        description="American english scene description for the image"
    )
    chinese_prompt: str = Field(
        description="Traditional Chinese translation of american english description"
    )


def _build_chain():
    model = _Model.build()
    prompt = _PromptTemplate.build()
    parser = PydanticOutputParser(pydantic_object=ImagePrompt)

    _chain = (
            {
                "user_input": itemgetter("user_input"),
                "format_instructions": lambda x: parser.get_format_instructions(),
            }
            | prompt
            | model
            | parser
    )

    return _chain


def run_image_prompt_generator_chain(user_input: str, model: str) -> ImagePrompt:
    _chain = _build_chain().with_config(configurable={"model": model})
    return _chain.invoke({"user_input": user_input})
