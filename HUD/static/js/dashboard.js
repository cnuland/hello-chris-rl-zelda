// VLM Vision Hybrid Dashboard - Real-time Updates

class Dashboard {
    constructor() {
        this.eventSource = null;
        this.lastUpdate = Date.now();
        this.init();
    }

    init() {
        console.log('🎮 Initializing VLM Vision Hybrid Dashboard');
        this.connectEventSource();
        this.startHeartbeat();
    }

    connectEventSource() {
        // Connect to Server-Sent Events stream
        this.eventSource = new EventSource('/stream');
        
        this.eventSource.onopen = () => {
            console.log('✅ Connected to training stream');
            this.updateConnectionStatus(true);
        };

        this.eventSource.onerror = () => {
            console.error('❌ Connection error');
            this.updateConnectionStatus(false);
            
            // Attempt reconnection after 3 seconds
            setTimeout(() => {
                if (this.eventSource.readyState === EventSource.CLOSED) {
                    console.log('🔄 Attempting reconnection...');
                    this.connectEventSource();
                }
            }, 3000);
        };

        this.eventSource.addEventListener('training_update', (event) => {
            const data = JSON.parse(event.data);
            this.handleTrainingUpdate(data);
        });

        this.eventSource.addEventListener('vision_update', (event) => {
            const data = JSON.parse(event.data);
            this.handleVisionUpdate(data);
        });
    }

    handleTrainingUpdate(data) {
        this.lastUpdate = Date.now();

        // Training Progress
        if (data.epoch !== undefined) {
            document.getElementById('epoch').textContent = data.epoch;
        }
        if (data.episode !== undefined) {
            document.getElementById('episode').textContent = data.episode;
        }
        if (data.episode_id !== undefined) {
            document.getElementById('episode-id').textContent = data.episode_id;
        }
        if (data.global_step !== undefined) {
            document.getElementById('global-step').textContent = data.global_step.toLocaleString();
        }
        if (data.episode_reward !== undefined) {
            document.getElementById('episode-reward').textContent = data.episode_reward.toFixed(1);
        }
        if (data.episode_length !== undefined) {
            document.getElementById('episode-length').textContent = data.episode_length;
        }

        // Game State
        if (data.location) {
            document.getElementById('location').textContent = data.location;
        }
        if (data.room_id !== undefined) {
            document.getElementById('room-id').textContent = data.room_id;
        }
        if (data.position) {
            document.getElementById('position').textContent = `(${data.position.x}, ${data.position.y})`;
        }
        if (data.health) {
            const hearts = '❤️'.repeat(data.health.current);
            const emptyHearts = '🖤'.repeat(data.health.max - data.health.current);
            document.getElementById('health').textContent = `${data.health.current}/${data.health.max} ${hearts}${emptyHearts}`;
        }

        // Entities
        if (data.entities) {
            document.getElementById('npc-count').textContent = data.entities.npcs || 0;
            document.getElementById('enemy-count').textContent = data.entities.enemies || 0;
            document.getElementById('item-count').textContent = data.entities.items || 0;
        }

        // LLM Guidance
        if (data.llm_suggestion) {
            document.getElementById('llm-suggestion').textContent = data.llm_suggestion;
            this.animateSuggestion();
        }
        if (data.llm_calls !== undefined) {
            document.getElementById('llm-calls').textContent = data.llm_calls;
        }
        if (data.llm_success_rate !== undefined) {
            document.getElementById('llm-success-rate').textContent = `${data.llm_success_rate.toFixed(1)}%`;
        }
        if (data.llm_alignment !== undefined) {
            document.getElementById('llm-alignment').textContent = data.llm_alignment;
        }

        // Milestones
        if (data.milestones) {
            this.updateMilestone('maku-tree', data.milestones.maku_tree_entered);
            this.updateMilestone('dungeon', data.milestones.dungeon_entered);
            if (data.milestones.sword_usage !== undefined) {
                const milestoneEl = document.getElementById('milestone-sword-use');
                milestoneEl.querySelector('.milestone-count').textContent = data.milestones.sword_usage;
            }
        }

        // Exploration
        if (data.exploration) {
            document.getElementById('rooms-discovered').textContent = data.exploration.rooms_discovered || 0;
            document.getElementById('grid-areas').textContent = data.exploration.grid_areas || 0;
            document.getElementById('buildings-entered').textContent = data.exploration.buildings_entered || 0;
        }
    }

    handleVisionUpdate(data) {
        // Update vision image
        if (data.image) {
            const visionImage = document.getElementById('vision-image');
            visionImage.src = `data:image/jpeg;base64,${data.image}`;
            
            // Hide loading overlay
            const overlay = document.getElementById('vision-loading');
            if (overlay) {
                overlay.classList.add('hidden');
            }

            // Update last updated timestamp
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
            document.getElementById('last-updated').textContent = timeStr;
        }

        // Update response time
        if (data.response_time !== undefined) {
            document.getElementById('response-time').textContent = `${data.response_time.toFixed(0)} ms`;
        }
    }

    updateMilestone(id, completed) {
        const milestoneEl = document.getElementById(`milestone-${id}`);
        if (!milestoneEl) return;

        const statusEl = milestoneEl.querySelector('.milestone-status');
        if (completed) {
            milestoneEl.classList.add('completed');
            statusEl.textContent = '✅';
        } else {
            milestoneEl.classList.remove('completed');
            statusEl.textContent = '❌';
        }
    }

    animateSuggestion() {
        const suggestionEl = document.getElementById('llm-suggestion');
        suggestionEl.style.animation = 'none';
        setTimeout(() => {
            suggestionEl.style.animation = 'pulse 0.5s ease-in-out';
        }, 10);
    }

    updateConnectionStatus(connected) {
        const badge = document.getElementById('status-badge');
        const statusText = badge.querySelector('.status-text');
        
        if (connected) {
            badge.style.borderColor = 'var(--green)';
            statusText.textContent = 'Connected';
            statusText.style.color = 'var(--green)';
        } else {
            badge.style.borderColor = 'var(--red)';
            statusText.textContent = 'Disconnected';
            statusText.style.color = 'var(--red)';
        }
    }

    startHeartbeat() {
        // Check for stale data every 5 seconds
        setInterval(() => {
            const timeSinceUpdate = Date.now() - this.lastUpdate;
            if (timeSinceUpdate > 10000) { // 10 seconds without update
                console.warn('⚠️ No updates received in 10 seconds');
                // Don't disconnect, but could show warning
            }
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 VLM Vision Hybrid Dashboard loaded');
    window.dashboard = new Dashboard();
});

