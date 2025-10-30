FROM python:3.14-slim

RUN pip install uv

WORKDIR /app

# CORRECT (if all files share a common pattern)
COPY *.py */*.toml */*.lock *.bin /app/

RUN uv sync 

EXPOSE 9696 

# RUN ls -la

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

# ENTRYPOINT [ "bash" ]
