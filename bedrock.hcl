version = "1.0"

train {
    step preprocess {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 preprocess.py"]}]
        resources {
            cpu = "2"
            memory = "12G"
        }
    }

    step train {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "2"
            memory = "12G"
        }
        depends_on = ["preprocess"]
    }

    parameters {
        PROJECT = "span-production"
        RAW_BUCKET = "bedrock-sample"
        RAW_DATA_DIR = "chestxray"
        BUCKET = "span-temp-production"
        BASE_DIR = "pneumonia"
        PREPROCESSED_DIR = "preprocessed"
    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]
    parameters {
        WORKERS = "1"
    }
}
