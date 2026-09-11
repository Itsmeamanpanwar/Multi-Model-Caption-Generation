# Multimodal Caption Generator

A Flask-based multimodal AI application that generates natural-language captions from both images and audio files using open-source deep learning models.

The project combines computer vision and audio understanding into a single web application, allowing users to upload an image or audio file and receive an automatically generated description.

## Features

- Image caption generation
- Audio caption generation
- Simple web-based interface
- REST API endpoints for image and audio captioning
- Local inference using open-source models
- Supports common image and audio formats
- Flask backend
- Responsive frontend
- Gunicorn-compatible production server
- No external paid AI API required

## Architecture

The application uses separate models for image and audio understanding.

```text
                    Multimodal Caption Generator
                              |
                    Flask Web Application
                              |
              +---------------+---------------+
              |                               |
        Image Input                     Audio Input
              |                               |
      BLIP Image Model               AudioCaps Model
              |                               |
      Image Understanding           Audio Understanding
              |                               |
              +---------------+---------------+
                              |
                       Generated Caption
