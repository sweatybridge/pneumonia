version = "1.0"

train {
    step train {
        image = "quay.io/basisai/python-cuda:3.9.2-10.1"
        install = [
            "apt-get update",
            "apt-get install -y git",
            "rm -rf /var/lib/apt/lists/*",
            "pip install -r /app/inhouse/requirements-train.txt"
        ]
        script = [{sh = [
            "cd /app/inhouse",
            "git clone --depth 1 https://github.com/ieee8023/covid-chestxray-dataset",
            "curl -o se_resnext50_32x4d-a260b3a4.pth https://storage.googleapis.com/bedrock-sample/chestxray/pytorch-se-resnext/se_resnext50_32x4d-a260b3a4.pth",
            "python train.py"
        ]}]
        resources {
            cpu = "3"
            memory = "15G"
            gpu = "1"
        }
    }

    parameters {
        TARGET_CLASS = "COVID-19"
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
        TARGET_CLASS = "COVID-19"
        WORKERS = "1"
    }
}
