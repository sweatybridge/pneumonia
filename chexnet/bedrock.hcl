version = "1.0"

serve {
    image = "quay.io/basisai/express-flask:v0.0.4-opencv"
    install = [
        "pip install -r /model-server/chexnet/requirements.txt"
    ]
    script = [{sh = [
        "cd /model-server/chexnet",
        "/app/entrypoint.sh"
    ]}]

    parameters {
        BEDROCK_SERVER = "serve"
        WORKERS = "2"
    }
}
