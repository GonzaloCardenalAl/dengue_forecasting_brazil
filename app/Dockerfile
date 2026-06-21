FROM python:3.13-slim
WORKDIR /repo
COPY . .
RUN pip install --no-cache-dir uv && uv sync --frozen --package dengue-app
RUN chmod +x app/docker-entrypoint.sh

ENV DENGUE_RUN_DIR=/repo/ml/results/production_run

EXPOSE 8000
ENTRYPOINT ["app/docker-entrypoint.sh"]
