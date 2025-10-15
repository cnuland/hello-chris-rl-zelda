"""
Session Manager for Zelda RL Training
Saves checkpoints and episode summaries to MinIO 'sessions' bucket
"""

import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any, Optional
import tempfile
from pathlib import Path


class SessionManager:
    """Manages saving training sessions (checkpoints + summaries) to MinIO."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize SessionManager with MinIO connection.
        
        Args:
            session_id: Unique identifier for this training session (default: timestamp)
        """
        # Generate session ID if not provided
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get MinIO credentials from environment
        self.s3_endpoint = os.environ.get('S3_ENDPOINT_URL')
        self.s3_access_key = os.environ.get('S3_ACCESS_KEY_ID')
        self.s3_secret_key = os.environ.get('S3_SECRET_ACCESS_KEY')
        self.s3_region = os.environ.get('S3_REGION_NAME', 'us-east-1')
        
        # Session bucket
        self.bucket_name = 'sessions'
        
        # Initialize S3 client if credentials available
        self.s3_client = None
        self.enabled = False
        
        if self.s3_endpoint and self.s3_access_key and self.s3_secret_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.s3_endpoint,
                    aws_access_key_id=self.s3_access_key,
                    aws_secret_access_key=self.s3_secret_key,
                    region_name=self.s3_region,
                    config=boto3.session.Config(signature_version='s3v4')
                )
                
                # Test connection and ensure bucket exists
                try:
                    self.s3_client.head_bucket(Bucket=self.bucket_name)
                    self.enabled = True
                    print(f"✅ SessionManager initialized: {self.session_id}")
                    print(f"   S3 Endpoint: {self.s3_endpoint}")
                    print(f"   Bucket: {self.bucket_name}")
                except:
                    print(f"⚠️  Bucket '{self.bucket_name}' not found, attempting to create...")
                    try:
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                        self.enabled = True
                        print(f"✅ Created bucket: {self.bucket_name}")
                    except Exception as e:
                        print(f"❌ Failed to create bucket: {e}")
                        self.enabled = False
                        
            except Exception as e:
                print(f"⚠️  SessionManager disabled: Failed to connect to S3: {e}")
                self.enabled = False
        else:
            print(f"⚠️  SessionManager disabled: Missing S3 credentials")
    
    def save_episode_summary(
        self,
        worker_id: int,
        episode_num: int,
        summary_data: Dict[str, Any]
    ) -> bool:
        """
        Save episode summary to MinIO.
        
        Args:
            worker_id: Worker/instance ID
            episode_num: Episode number
            summary_data: Dictionary with episode metrics and data
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Add metadata
            summary_data['session_id'] = self.session_id
            summary_data['worker_id'] = worker_id
            summary_data['episode_num'] = episode_num
            summary_data['timestamp'] = datetime.now().isoformat()
            
            # Create S3 key path
            s3_key = f"{self.session_id}/worker_{worker_id}/episode_{episode_num:06d}_summary.json"
            
            # Convert to JSON
            json_data = json.dumps(summary_data, indent=2)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"💾 Saved episode {episode_num} summary: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save episode summary: {e}")
            return False
    
    def save_checkpoint(
        self,
        worker_id: int,
        episode_num: int,
        checkpoint_data: Dict[str, Any],
        model_state: Optional[bytes] = None
    ) -> bool:
        """
        Save training checkpoint to MinIO.
        
        Args:
            worker_id: Worker/instance ID
            episode_num: Episode number
            checkpoint_data: Dictionary with checkpoint metadata
            model_state: Optional serialized model weights
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Add metadata
            checkpoint_data['session_id'] = self.session_id
            checkpoint_data['worker_id'] = worker_id
            checkpoint_data['episode_num'] = episode_num
            checkpoint_data['timestamp'] = datetime.now().isoformat()
            
            # Save checkpoint metadata
            s3_key_meta = f"{self.session_id}/worker_{worker_id}/episode_{episode_num:06d}_checkpoint.json"
            json_data = json.dumps(checkpoint_data, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key_meta,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            # Save model weights if provided
            if model_state:
                s3_key_model = f"{self.session_id}/worker_{worker_id}/episode_{episode_num:06d}_model.pth"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key_model,
                    Body=model_state,
                    ContentType='application/octet-stream'
                )
                print(f"💾 Saved checkpoint {episode_num}: s3://{self.bucket_name}/{s3_key_meta}")
                print(f"   Model weights: s3://{self.bucket_name}/{s3_key_model}")
            else:
                print(f"💾 Saved checkpoint {episode_num}: s3://{self.bucket_name}/{s3_key_meta}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def save_session_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save overall session metadata.
        
        Args:
            metadata: Session-level metadata (config, hyperparameters, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            metadata['session_id'] = self.session_id
            metadata['timestamp'] = datetime.now().isoformat()
            
            s3_key = f"{self.session_id}/session_metadata.json"
            json_data = json.dumps(metadata, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"💾 Saved session metadata: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save session metadata: {e}")
            return False

