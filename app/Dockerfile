FROM python:3.13-slim
WORKDIR /repo
COPY . .
RUN pip install --no-cache-dir uv && uv sync --frozen --package dengue-app
EXPOSE 8000
CMD ["uv", "run", "--package", "dengue-app", "uvicorn", "dengue_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
