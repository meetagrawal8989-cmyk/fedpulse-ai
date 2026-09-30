#!/bin/bash
# ==============================================================================
# Automated Deployment Script for Google Cloud Run
# FedPulse AI Platform
# ==============================================================================

set -e

# Prompt for GCP Project ID if not set
if [ -z "$GCP_PROJECT" ]; then
    echo "🔍 Fetching active Google Cloud Project ID..."
    GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
fi

if [ -z "$GCP_PROJECT" ]; then
    echo "❌ Error: GCP_PROJECT is not set and no active gcloud project found."
    echo "Please set your project ID: export GCP_PROJECT='your-gcp-project-id'"
    exit 1
fi

REGION="us-central1"
SERVICE_NAME="fedpulse-ai"
IMAGE_NAME="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}:latest"

echo "🚀 Deploying FedPulse AI to Google Cloud Run..."
echo "------------------------------------------------"
echo "Project ID: ${GCP_PROJECT}"
echo "Region:     ${REGION}"
echo "Service:    ${SERVICE_NAME}"
echo "Image:      ${IMAGE_NAME}"
echo "------------------------------------------------"

# Step 1: Enable required GCP services
echo "📦 Step 1: Enabling Cloud Build & Cloud Run APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com --project="${GCP_PROJECT}"

# Step 2: Build container image using Google Cloud Build
echo "🏗️ Step 2: Building container image in GCP Cloud Build..."
gcloud builds submit --tag "${IMAGE_NAME}" --project="${GCP_PROJECT}" .

# Step 3: Deploy to Cloud Run
echo "🌩️ Step 3: Deploying container to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --project="${GCP_PROJECT}"

echo "------------------------------------------------"
echo "✅ Deployment Successful!"
echo "Your live public web app URL can be retrieved via:"
echo "gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)'"
