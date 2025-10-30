FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app 

COPY main.py pipeline_v1.bin pyproject.toml uv.lock ./

RUN uv sync

EXPOSE 8000

CMD ["uv","run","uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]

#ENTRYPOINT ["bash"]
