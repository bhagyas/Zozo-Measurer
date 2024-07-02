# Stage 1: Build environment
FROM python:3.8-slim AS builder

WORKDIR /usr/src/app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.8-slim

WORKDIR /usr/src/app

COPY --from=builder /usr/src/app /usr/src/app


# Install cv2 module
RUN pip install opencv-python-headless

# Copy all sources
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME ZozoMeasurer

# Mount output directory
VOLUME /usr/src/app/output

# Run detect_points.py when the container launches
# Note: This container expects an image file as an argument to detect_points.py
# CMD ["python3", "detect_points.py", "your_image.jpg"]

# Run above with the argument provided with docker run command
# CMD ["python3", "detect_points.py"]

# Entrypoint
ENTRYPOINT ["python3", "detect_points.py"]
