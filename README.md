The deployment of automated image captioning systems on edge devices is often hindered by the high computational demands of state-of-the-art Deep Learning models. 
This project presents a lightweight, image-to-text translation framework designed specifically for resource-constrained environments using the Flickr8k dataset.
By integrating a MobileNetV2-based encoder—characterized by inverted residual blocks—with an LSTM-driven sequence processor, the proposed system achieves a 1280-dimensional
feature extraction with significantly reduced floating-point operations (FLOPs). A "Merge" architecture is utilized to facilitate late fusion of visual and linguistic features, 
ensuring the model remains computationally efficient during inference. Experimental results demonstrate that implementing Beam Search (k=5) with length normalization yields 
a peak BLEU-4 score of 0.1631. Notably, the architecture utilizes only 3.5 million parameters, representing an 85% reduction in model size compared to standard InceptionV3-based
solutions, making it a viable candidate for real-time applications on mobile and IoT hardware.

This project presents an image-to-text translation model trained on the Flickr8k dataset. The architecture employs a Convolutional Neural Network (CNN), specifically MobileNetV2,
as an image feature extractor, coupled with a Long Short-Term Memory (LSTM) network for sequence processing. Late Fusion architecture is used to merge the vectored outputs. 
To enhance the syntactic and semantic quality of the generated captions, a Beam Search decoding algorithm is implemented with length normalization. The model is evaluated using
the Bilingual Evaluation Understudy (BLEU) metric across various beam widths (k=1, 3, and 5) to determine the optimal decoding strategy.


Moving from Greedy Search (k=1) to Beam Search (k=3,5)  yielded a significant performance boost. Specifically, 
the BLEU-4 score improved by approximately 18.61% (from 0.1375 to 0.1631). This validates that considering multiple 
candidate sequences allows the model to better handle long-range dependencies and produce more fluent sentences.

Our model achieves a BLEU-4 score of 0.1631. While heavier state-of-the-art models (such as InceptionV3-based architectures )often reach scores of 0.18–0.20 on Flickr8k, 
they utilize encoders with ~24 million parameters. In contrast, our MobileNetV2 encoder utilizes only 3.5 million parameters, making it approximately 85% lighter. 
This result demonstrates that our approach successfully generates intelligible captions while remaining lightweight enough for potential deployment on mobile or edge devices.

