from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class ImageCaptioner:

    def __init__(self):
        print("Loading image captioning model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        print("Model loaded successfully!")

    def generate_caption(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        output = self.model.generate(
            **inputs,
            max_new_tokens=50
        )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption