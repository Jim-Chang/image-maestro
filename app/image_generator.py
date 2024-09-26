import enum

import fal_client

from app.models import FalAiModel

MAX_PIXELS = 1440


class ImageSize(str, enum.Enum):
    square = "square"
    portrait_4_3 = "portrait_4_3"
    portrait_16_9 = "portrait_16_9"
    landscape_4_3 = "landscape_4_3"
    landscape_16_9 = "landscape_16_9"


def get_image_size(image_size: ImageSize):
    if image_size == ImageSize.square:
        return {"width": MAX_PIXELS, "height": MAX_PIXELS}
    elif image_size == ImageSize.portrait_4_3:
        return {"width": int(MAX_PIXELS * 3 / 4), "height": MAX_PIXELS}
    elif image_size == ImageSize.portrait_16_9:
        return {"width": int(MAX_PIXELS * 9 / 16), "height": MAX_PIXELS}
    elif image_size == ImageSize.landscape_4_3:
        return {"width": MAX_PIXELS, "height": int(MAX_PIXELS * 3 / 4)}
    elif image_size == ImageSize.landscape_16_9:
        return {"width": MAX_PIXELS, "height": int(MAX_PIXELS * 9 / 16)}


def generate_image(
        prompt: str,
        model: FalAiModel = FalAiModel.flux_schnell,
        image_size: ImageSize = ImageSize.landscape_4_3,
        num_inference_steps: int = 35,
        guidance_scale: float = 3.5,
        num_images: int = 1,
):
    arguments = {
        "prompt": prompt,
        "image_size": get_image_size(image_size),
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "safety_tolerance": 6,
    }

    handler = fal_client.submit(
        model,
        arguments=arguments,
    )

    result = handler.get()
    image_urls = [data["url"] for data in result["images"]]
    return image_urls
