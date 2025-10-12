# HUD Dashboard Kubernetes Deployment

The VLM Vision Hybrid HUD Dashboard provides real-time visualization of Zelda RL training progress with:
- 🎨 Purple neon vLLM branding
- 📸 Live view of game screen being analyzed by Vision LLM
- 📊 Real-time training metrics (rewards, episode length, etc.)
- 🔄 Server-Sent Events for low-latency updates
- 🔒 Session management (prevents multiple training sessions from conflicting)

## Architecture

### Components

1. **HUD Server** (`HUD/hud_server.py`)
   - Flask web server with SSE support
   - Serves dashboard UI and API endpoints
   - Manages training session registration
   - Runs on port 8086

2. **HUD Client** (`HUD/hud_client.py`)
   - Python client library for distributed training
   - Makes HTTP requests to HUD server
   - Handles session registration and updates
   - Used by Ray workers to send metrics

3. **Ray Callback** (`ray_hud_callback.py`)
   - Ray RLlib callback integration
   - Automatically sends training metrics to HUD
   - Captures episode rewards, lengths, and custom metrics
   - Sends vision data when available

### Kubernetes Resources

- **Deployment:** `ops/openshift/hud-deployment.yaml`
  - Single replica of HUD server
  - 0.5-1 CPU, 512Mi-1Gi memory
  - Health/readiness probes on `/health`
  
- **Service:** `zelda-hud-service`
  - ClusterIP service on port 8086
  - Internal cluster access for training workers
  
- **Route:** `zelda-hud-route`
  - External HTTPS access for browser viewing
  - Edge TLS termination

## Deployment

### From Notebook (Automated)

Cell [9] in `run-kuberay-zelda.ipynb` automatically deploys the HUD:

```python
!oc apply -f ops/openshift/hud-deployment.yaml
!oc wait --for=condition=ready pod -l app=zelda-hud -n zelda-hybrid-rl-llm --timeout=120s
```

### Manual Deployment

```bash
# Apply HUD deployment
oc apply -f ops/openshift/hud-deployment.yaml

# Wait for pod to be ready
oc wait --for=condition=ready pod -l app=zelda-hud -n zelda-hybrid-rl-llm --timeout=120s

# Get external URL
oc get route zelda-hud-route -n zelda-hybrid-rl-llm
```

## Usage

### Environment Variable

Training jobs must set the `HUD_URL` environment variable:

```python
env_vars = {
    'HUD_URL': 'http://zelda-hud-service.zelda-hybrid-rl-llm.svc.cluster.local:8086',
    # ... other vars ...
}
```

### Access Dashboard

1. **External URL (browser):**
   ```
   https://zelda-hud-route-zelda-hybrid-rl-llm.apps...openshiftapps.com
   ```

2. **Internal URL (training code):**
   ```
   http://zelda-hud-service.zelda-hybrid-rl-llm.svc.cluster.local:8086
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/stream` | GET | Server-Sent Events stream |
| `/api/register_session` | POST | Register training session |
| `/api/update_training` | POST | Update training metrics |
| `/api/update_vision` | POST | Update vision image |
| `/health` | GET | Health check for K8s probes |

## Session Management

Only **one training session** can update the HUD at a time:

1. Training job calls `/api/register_session`
2. Server returns `session_id` if no active session
3. All updates must include `session_id`
4. Session times out after 30 seconds of inactivity
5. New sessions can take over after timeout

This prevents conflicts when multiple Ray workers try to update simultaneously.

## Monitoring

### Check HUD Pod Status

```bash
oc get pods -l app=zelda-hud -n zelda-hybrid-rl-llm
oc logs -f deployment/zelda-hud -n zelda-hybrid-rl-llm
```

### Check HUD Route

```bash
oc get route zelda-hud-route -n zelda-hybrid-rl-llm
```

### Test Health Endpoint

```bash
curl http://zelda-hud-service.zelda-hybrid-rl-llm.svc.cluster.local:8086/health
```

## Troubleshooting

### HUD Not Updating

1. Check HUD pod is running:
   ```bash
   oc get pods -l app=zelda-hud -n zelda-hybrid-rl-llm
   ```

2. Check HUD logs:
   ```bash
   oc logs -f deployment/zelda-hud -n zelda-hybrid-rl-llm
   ```

3. Verify training job has `HUD_URL` environment variable set

4. Check if another session is active (check HUD logs for session registration messages)

### Cannot Access Dashboard in Browser

1. Get route URL:
   ```bash
   oc get route zelda-hud-route -n zelda-hybrid-rl-llm -o jsonpath='{.spec.host}'
   ```

2. Verify route is created and has a host

3. Check if you're logged into OpenShift (browser auth required for external route)

### Training Job Can't Connect to HUD

1. Verify service exists:
   ```bash
   oc get svc zelda-hud-service -n zelda-hybrid-rl-llm
   ```

2. Check service endpoints:
   ```bash
   oc get endpoints zelda-hud-service -n zelda-hybrid-rl-llm
   ```

3. Verify HUD pod is ready:
   ```bash
   oc get pods -l app=zelda-hud -n zelda-hybrid-rl-llm
   ```

## Development

### Local Testing

```bash
cd HUD
python hud_server.py
```

Access at: `http://localhost:8086`

### Test HUD Client

```python
from HUD.hud_client import HUDClient

client = HUDClient(hud_url='http://localhost:8086')
if client.enabled:
    client.update_training_data({
        'episode_reward': 100.0,
        'episode_length': 500,
    })
```

## Features

### Real-Time Metrics

- Episode rewards and lengths
- Total timesteps and episodes
- Mean rewards over time
- Policy loss, value loss, entropy
- Learning rate
- Custom metrics from environment

### Vision Display

- Latest game screenshot being analyzed by LLM
- LLM response time
- Updated every LLM call (configurable frequency)

### Single Episode Viewer

**Important:** With 9 parallel training environments (3 workers × 3 envs each), only **ONE designated environment** sends data to the HUD to avoid confusion.

- **Designated Reporter:** Worker 1, Environment 0
- **Why:** Prevents rapid switching between different episodes
- **Result:** Viewer sees ONE consistent episode throughout training
- **Note:** All 9 environments still train; only display data comes from one

To change which environment is viewable, modify in `ray_hud_callback.py`:
```python
HUD_WORKER_INDEX = 1  # First rollout worker
HUD_ENV_ID = 0        # First environment on that worker
```

### UI Features

- Auto-scrolling metrics table
- Large vision image display
- Color-coded status indicators
- Responsive design
- vLLM branding and logo

