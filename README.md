# Zozo-Measurer
Track the markers on the Zozo-suit with OpenCV.

## Installation
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running
Run `detect_points.py` with a filename as first argument.

Example:
```bash
python3 detect_points.py ./data/IMG_20180911_163153_c.jpg
```

Runnig the script will create two files:

* `point_positions.png` The original image with
positions, IDs and distances drawn into the image.
* `collected_points.png` A usually very wide image
with all points in their original skewed position,
the corrected version and an image with a drawing
of the point it detected. Below is the ID, the
confidence score in percent and the calculated distance.

## Using in your own code
If you want to use `detect_points.py` in your own code,
import `detect_points`. The function does all the heavy
lifting and returns IDs, confidence scores (0 to 1),
positions and distances. If you need raw data from the
recognition use the `raw_data` dictionary which is returned
as the last variable.


## Running using Docker (Headless/non-window mode)

```bash
# Build the image
docker build -t zozo-measurer .

#Mount the /data and /output directories to the container. Add the file you want to process as the last argument.
docker run -v $(pwd)/./output:/usr/src/app/output -v $(pwd)/./data:/usr/src/app/data zozo-measurer ./data/IMG_20180911_163153_c.jpg

# The output will be in the output directory.
```

# Contributors
Bhagya Nirmaan Silva - [bhagyas](github.com/bhagyas)
