# Use an official Python base image
FROM python:3.11-slim

# Install system dependencies required for geospatial packages (Rasterio, Fiona, GDAL, GeoPandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Upgrade pip and install the project in editable mode with dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Expose the port that Streamlit runs on
EXPOSE 8501

# Default command: Run the Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]