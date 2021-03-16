version = "1.0"

serve {
    image = "quay.io/basisai/express-flask:v0.0.4-opencv"
    install = [
        "pip install -r requirements.txt"
    ]
    script = [
        {sh = ["/app/entrypoint.sh"]}
    ]

    parameters {
        BEDROCK_SERVER = "serve"
        WORKERS = "2"
    }
}
