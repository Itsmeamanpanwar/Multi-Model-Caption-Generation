from flask import Blueprint, request, jsonify
from app.services.image_caption import ImageCaptioner
from app.services.audio_caption import AudioCaptioner
import os


caption_routes = Blueprint("caption_routes", __name__)


# Load models once when Flask starts
image_captioner = ImageCaptioner()
audio_captioner = AudioCaptioner()


@caption_routes.route("/caption/image", methods=["POST"])
def caption_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No image selected"}), 400

    os.makedirs("outputs", exist_ok=True)

    image_path = os.path.join("outputs", image.filename)

    image.save(image_path)

    caption = image_captioner.generate_caption(image_path)

    return jsonify({"caption": caption})


@caption_routes.route("/caption/audio", methods=["POST"])
def caption_audio():

    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio = request.files["audio"]

    if audio.filename == "":
        return jsonify({"error": "No audio selected"}), 400

    os.makedirs("outputs", exist_ok=True)

    audio_path = os.path.join("outputs", audio.filename)

    audio.save(audio_path)

    caption = audio_captioner.generate_caption(audio_path)

    return jsonify({"caption": caption})