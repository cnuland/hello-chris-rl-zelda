# HUD Container Build and Deployment Guide

This directory contains the VLM Vision Hybrid HUD Dashboard Server - a lightweight Flask application for real-time training visualization.

## 📦 Container Image

**Image:** `quay.io/cnuland/zelda-hud:latest`

**Base:** Python 3.11 slim (much smaller than Ray worker image)

**Size:** ~150MB (vs ~5GB for Ray worker image)

---

## 🔨 Building the Container

### Quick Build (with script)

```bash
cd HUD
./build-container.sh [version]
```

The script will:
1. Build the container image
2. Show image details
3. Prompt to push to registry

**Examples:**
```bash
./build-container.sh           # Build with 'latest' tag
./build-container.sh v1.0.0    # Build with specific version
```

### Manual Build

```bash
cd HUD

# Build with Podman
podman build -t quay.io/cnuland/zelda-hud:latest -f Containerfile .

# Or with Docker
docker build -t quay.io/cnuland/zelda-hud:latest -f Containerfile .

# Push to registry
podman push quay.io/cnuland/zelda-hud:latest
```

---

## 🧪 Testing Locally

### Run the container

```bash
podman run -p 8086:8086 quay.io/cnuland/zelda-hud:latest
```

### Access the dashboard

Open in browser: http://localhost:8086

You should see the purple neon vLLM dashboard with "Waiting for training session..."

### Test with HUD client

```python
from HUD.hud_client import HUDClient

client = HUDClient(hud_url='http://localhost:8086')
if client.enabled:
    client.update_training_data({
        'episode_reward': 100.0,
        'episode_length': 500,
        'timesteps_total': 1000,
    })
```

---

## 🚀 Deploying to Kubernetes/OpenShift

### 1. Build and push the image

```bash
cd HUD
./build-container.sh
# Answer 'y' when prompted to push
```

### 2. Deploy to cluster

```bash
# From project root
oc apply -f ops/openshift/hud-deployment.yaml
```

### 3. Wait for pod to be ready

```bash
oc wait --for=condition=ready pod -l app=zelda-hud -n zelda-hybrid-rl-llm --timeout=120s
```

### 4. Get the dashboard URL

```bash
oc get route zelda-hud-route -n zelda-hybrid-rl-llm -o jsonpath='{.spec.host}'
```

---

## 📁 Container Contents

```
/app/
├── hud_server.py          # Flask server
├── templates/
│   └── index.html         # Dashboard HTML
├── static/
│   ├── css/
│   │   └── style.css      # Purple neon styles
│   ├── js/
│   │   └── dashboard.js   # SSE client
│   └── images/
│       └── vllm-logo-text-light.png
└── requirements.txt       # Python dependencies
```

---

## 🔧 Container Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |

### Ports

- **8086:** HTTP server (Flask)

### Health Check

- **Endpoint:** `/health`
- **Interval:** 30s
- **Timeout:** 10s
- **Retries:** 3

---

## 📊 Dependencies

All dependencies are minimal and Flask-specific:

```
Flask==3.0.0           # Web framework
Flask-CORS==4.0.0      # CORS support
Werkzeug==3.0.1        # WSGI utilities
waitress==2.1.2        # Production WSGI server
requests==2.31.0       # HTTP client (for health check)
```

**Total size:** ~30MB of Python packages

---

## 🔄 Updating the Container

When you make changes to the HUD:

```bash
cd HUD

# Rebuild with new version
./build-container.sh v1.1.0

# Update deployment (optional - use specific version)
oc set image deployment/zelda-hud \
  hud-server=quay.io/cnuland/zelda-hud:v1.1.0 \
  -n zelda-hybrid-rl-llm

# Or just delete pod to pull latest
oc delete pod -l app=zelda-hud -n zelda-hybrid-rl-llm
```

---

## 🐛 Troubleshooting

### Container won't build

```bash
# Check Containerfile syntax
podman build --no-cache -t test -f Containerfile .

# Check if requirements.txt is valid
cat requirements.txt
```

### Container won't start

```bash
# Check logs
podman logs <container-id>

# Run interactively
podman run -it --entrypoint /bin/bash quay.io/cnuland/zelda-hud:latest
```

### Image won't push

```bash
# Login to registry
podman login quay.io

# Check image exists
podman images | grep zelda-hud

# Try manual push
podman push quay.io/cnuland/zelda-hud:latest
```

### Pod won't start in K8s

```bash
# Check pod events
oc describe pod -l app=zelda-hud -n zelda-hybrid-rl-llm

# Check logs
oc logs -l app=zelda-hud -n zelda-hybrid-rl-llm

# Check if image exists and is accessible
oc get events -n zelda-hybrid-rl-llm | grep ImagePull
```

---

## 🎯 Why a Separate Container?

**Before:** Using Ray worker image
- ❌ 5GB+ image size
- ❌ Includes PyTorch, Ray, CUDA, etc.
- ❌ HUD tied to Ray infrastructure
- ❌ Slow to pull and deploy

**After:** Dedicated HUD image
- ✅ ~150MB image size (30x smaller!)
- ✅ Only Flask and minimal dependencies
- ✅ Independent from training infrastructure
- ✅ Fast to pull and deploy
- ✅ Can version HUD separately

---

## 📖 Related Documentation

- **Kubernetes Deployment:** [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md)
- **HUD Server Code:** [hud_server.py](hud_server.py)
- **HUD Client:** [hud_client.py](hud_client.py)
- **Main Deployment:** [../ops/openshift/hud-deployment.yaml](../ops/openshift/hud-deployment.yaml)

