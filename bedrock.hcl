version = "1.0"

train {
    step train {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
            "pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "2"
            memory = "12G"
        }
    }

    parameters {
        BASE_DIR = "chestxray/"
    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
        "pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
    ]
    script = [
        {sh = [
            "gunicorn --config gunicorn_config.py --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]

    parameters {
        WORKERS = "1"
        prometheus_multiproc_dir = "/tmp"
    }
}
