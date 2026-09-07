import torch
import torchaudio

from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download


class AudioCaptioner:

    def __init__(self):

        print("Loading audio captioning model...")

        self.device = torch.device("cpu")

        model_name = "wsntxxn/effb2-trm-audiocaps-captioning"

        # Load configuration
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Construct model without from_pretrained()
        self.model = AutoModel.from_config(
            config,
            trust_remote_code=True
        )

        # Download/load weights manually
        weights_path = hf_hub_download(
            model_name,
            "pytorch_model.bin"
        )

        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "wsntxxn/audiocaps-simple-tokenizer"
        )

        print("Audio captioning model loaded successfully!")

    def generate_caption(self, audio_path):

        wav, sample_rate = torchaudio.load(audio_path)

        wav = torchaudio.functional.resample(
            wav,
            sample_rate,
            self.model.config.sample_rate
        )

        if wav.size(0) > 1:
            wav = wav.mean(dim=0).unsqueeze(0)

        audio_length = [wav.size(1)]

        wav = wav.to(self.device)

        with torch.no_grad():

            output = self.model(
                audio=wav,
                audio_length=audio_length
            )

        caption = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption
