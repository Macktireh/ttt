FROM docker.io/ollama/ollama:0.9.0

ARG APP_HOME=/ollama

WORKDIR $APP_HOME

RUN ollama serve & \
    sleep 5 && \
    ollama pull deepseek-r1:7b && \
    ollama pull qwen3:4b && \
    pkill ollama

EXPOSE 11434
