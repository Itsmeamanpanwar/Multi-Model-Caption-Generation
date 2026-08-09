import gradio as gr
from tensorflow.keras.models import load_model
from predict import load_tokenizer, extract_features, beam_search_predictions, MAX_LENGTH, BEAM_WIDTH

print("Loading tokenizer and caption model...")
tokenizer = load_tokenizer("tokenizer.json")
caption_model = load_model("image_caption_model.keras")
print("Model loaded. Ready to serve requests.")


def generate_caption(pil_image):
    if pil_image is None:
        return "Please upload an image first."

    # Save the uploaded PIL image to a temp file since extract_features()
    # expects a filepath (same as it did in the Flask version).
    temp_path = "temp_upload.jpg"
    pil_image.convert("RGB").save(temp_path)

    features = extract_features(temp_path)
    caption = beam_search_predictions(
        caption_model, features, tokenizer,
        max_length=MAX_LENGTH, beam_index=BEAM_WIDTH
    )
    return caption


demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Caption Generator",
    description="Upload a photo. A MobileNetV2 + LSTM model (beam search) will describe what's in it.",
)

if __name__ == "__main__":
    demo.launch()
