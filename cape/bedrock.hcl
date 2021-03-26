version = "1.0"

serve {
    image = "quay.io/basisai/express-flask:v0.0.4"
    install = [
        "apt-get update",
        "apt-get install -y libopencv-dev",
        "rm -rf /var/lib/apt/lists/*",
        "pip install -r /model-server/cape/requirements.txt"
    ]
    script = [{sh = [
        "cd /model-server/cape",
        "/app/entrypoint.sh"
    ]}]

    parameters {
        BEDROCK_SERVER = "serve"
        WORKERS = "2"
    }
}
