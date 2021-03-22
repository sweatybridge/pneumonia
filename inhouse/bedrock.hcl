version = "1.0"

train {
    step train {
        image = "quay.io/basisai/python-cuda:3.9.2-10.1"
        install = [
            "pip install -r /app/inhouse/requirements-train.txt"
        ]
        script = [{sh = [
            "cd /app/inhouse",
            "python train.py"
        ]}]
        resources {
            cpu = "2"
            memory = "12G"
            gpu = "1"
        }
    }

    parameters {
        PROJECT = "span-production"
        RAW_BUCKET = "bedrock-sample"
        RAW_DATA_DIR = "chestxray"
    }
}

serve {
    image = "quay.io/basisai/python-cuda:3.9.2-10.1"
    install = [
        "pip install -r /model-server/inhouse/requirements-serve.txt"
    ]
    script = [
        {sh = [
            "cd /model-server/inhouse",
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]
    parameters {
        WORKERS = "1"
    }
}
