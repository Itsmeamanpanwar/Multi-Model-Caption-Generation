from app.services.image_caption import ImageCaptioner


image_path = "data/images/test.jpg"

captioner = ImageCaptioner()

caption = captioner.generate_caption(image_path)

print("\nGenerated Caption:")
print(caption)