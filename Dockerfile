# Stage 1 of the plan in DOCKER.md: containerise the application exactly as it
# runs today. One service, no database, no model. Prove this before adding
# anything else.
#
# Two targets:
#   base  -> the application image (default)
#   test  -> base + pytest, so the suite runs inside the container

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before source. Editing a .py file then does not invalidate the
# pip layer, so rebuilds take seconds instead of minutes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a normal user, not root. Streamlit writes into $HOME, so give it one.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8501

# Streamlit's own health endpoint. Verified working before this file was written.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# --server.address=0.0.0.0 is mandatory. Streamlit binds to 127.0.0.1 by
# default, which inside a container means unreachable from the host.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]


# ---------------------------------------------------------------------------
# Test image. Same code, same Python, plus pytest.
# Exit criterion for stage 1: this must report 179 passed.
# ---------------------------------------------------------------------------
FROM base AS test

USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
USER appuser

CMD ["pytest", "-q"]
