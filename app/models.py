import enum


class GoogleModel(str, enum.Enum):
    gemini_15_pro_latest = "gemini-1.5-pro-latest"
    gemini_15_pro_exp_0801 = "gemini-1.5-pro-exp-0801"
    gemini_15_pro_exp_0827 = "gemini-1.5-pro-exp-0827"


class AnthropicModel:
    claude_3_opus_20240229 = "claude-3-opus-20240229"
    claude_3_sonnet_20240229 = "claude-3-sonnet-20240229"
    claude_3_haiku_20240307 = "claude-3-haiku-20240307"

    claude_3_5_sonnet_20240620 = "claude-3-5-sonnet-20240620"


class FalAiModel(str, enum.Enum):
    flux_pro = "fal-ai/flux-pro"
    flux_dev = "fal-ai/flux/dev"
    flux_schnell = "fal-ai/flux/schnell"

    flux_realism = "fal-ai/flux-realism"
