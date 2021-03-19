import json
from logging import getLogger

from flask import Flask, Response, current_app, request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from werkzeug.exceptions import HTTPException

from serve import Model

logger = getLogger()
app = Flask(__name__)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({"type": e.name, "reason": e.description})
    response.content_type = "application/json"
    return response


@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate the model monitoring service here instead.
    """
    current_app.model = Model(logger=logger)


@app.route("/", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    features = current_app.model.pre_process(
        http_body=request.data, files=request.files
    )
    return current_app.model.predict(features=features)[0]
    # return current_app.model.post_process(score=score, prediction_id=None)


@app.route("/explain/", defaults={"target": None}, methods=["POST"])
@app.route("/explain/<target>", methods=["POST"])
def explain(target):
    features = current_app.model.pre_process(
        http_body=request.data, files=request.files
    )
    return current_app.model.explain(features=features, target=target)[0]


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run()
