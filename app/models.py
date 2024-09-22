import enum


class GoogleModel(str, enum.Enum):
    gemini_15_pro_latest = "gemini-1.5-pro-latest"
    gemini_15_pro_exp_0801 = "gemini-1.5-pro-exp-0801"
    gemini_15_pro_exp_0827 = "gemini-1.5-pro-exp-0827"


class FalAiModel(str, enum.Enum):
    flux_pro = "fal-ai/flux-pro"
    flux_dev = "fal-ai/flux/dev"
    flux_schnell = "fal-ai/flux/schnell"

    flux_realism = "fal-ai/flux-realism"
