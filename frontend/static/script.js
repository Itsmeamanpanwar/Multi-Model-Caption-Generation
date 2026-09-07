async function generateCaption() {

    const input = document.getElementById("imageInput");
    const caption = document.getElementById("caption");
    const preview = document.getElementById("preview");

    if (input.files.length === 0) {
        caption.textContent = "Please select an image.";
        return;
    }

    const image = input.files[0];

    const imageURL = URL.createObjectURL(image);

    preview.innerHTML = `
        <img src="${imageURL}" alt="Selected image">
    `;

    caption.textContent = "Generating image caption...";

    const formData = new FormData();

    formData.append("image", image);

    try {

        const response = await fetch("/caption/image", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            caption.textContent = data.caption;
        } else {
            caption.textContent = data.error;
        }

    } catch (error) {

        caption.textContent = "Something went wrong.";
        console.error(error);

    }
}


async function generateAudioCaption() {

    const input = document.getElementById("audioInput");
    const caption = document.getElementById("caption");

    if (input.files.length === 0) {
        caption.textContent = "Please select an audio file.";
        return;
    }

    const audio = input.files[0];

    caption.textContent = "Generating audio caption...";

    const formData = new FormData();

    formData.append("audio", audio);

    try {

        const response = await fetch("/caption/audio", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            caption.textContent = data.caption;
        } else {
            caption.textContent = data.error;
        }

    } catch (error) {

        caption.textContent = "Something went wrong.";
        console.error(error);

    }
}