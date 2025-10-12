#!/bin/bash

# AWS Configuration Script
# This script sets up your AWS environment securely

export AWS_ACCESS_KEY_ID="AKIARIZO57HKNTFQ5F5I"
export AWS_SECRET_ACCESS_KEY="DyBqPQjd5L2dhaQDZoDKQzAshAM7KndNZM13S2j"
export AWS_DEFAULT_REGION="eu-west-1"

# AWS Console Details (for reference)
# Console URL: https://087608392148.signin.aws.amazon.com/console
# Username: cnuland@redhat.com-9p74m-admin
# Password: HpNIkyEqi7yb

# Bastion Host Details
export BASTION_IP="54.171.192.145"

echo "AWS Configuration loaded successfully!"
echo "Account ID: 087608392148"
echo "Region: eu-west-1"
echo "Bastion IP: $BASTION_IP"

# Quick status check
echo ""
echo "=== AWS Status Check ==="
aws sts get-caller-identity --query 'Account' --output text
echo ""
echo "=== EC2 Instances ==="
aws ec2 describe-instances --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,Tags[?Key==`Name`].Value|[0]]' --output table