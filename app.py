from flask import Flask, render_template
from app.routes.caption_routes import caption_routes


app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

app.register_blueprint(caption_routes)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)