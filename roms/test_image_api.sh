#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path> [prompt]"
    echo "Example: $0 ~/Downloads/image.jpg 'What do you see in this image?'"
    exit 1
fi

IMAGE_PATH="$1"
PROMPT="${2:-What do you see in this image?}"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Get file extension to determine MIME type
EXT="${IMAGE_PATH##*.}"
case "$EXT" in
    jpg|jpeg)
        MIME_TYPE="image/jpeg"
        ;;
    png)
        MIME_TYPE="image/png"
        ;;
    *)
        MIME_TYPE="image/jpeg"
        ;;
esac

BASE64_IMAGE=$(base64 -i "$IMAGE_PATH")

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user", 
        "content": "'"$PROMPT"'",
        "images": ["data:'"$MIME_TYPE"';base64,'"$BASE64_IMAGE"'"]
      }
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | python3 -m json.tool

