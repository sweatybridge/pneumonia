#!/bin/sh

mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
maxUploadSize = 10\n\
" > ~/.streamlit/config.toml
