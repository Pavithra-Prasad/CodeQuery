#!/bin/bash
# CodeQuery AI — Google Cloud Run Deployment Script
# Usage: ./deploy.sh

set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SERVICE_NAME="codequery-ai"
REGION="us-central1"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "🚀 Deploying CodeQuery AI to Cloud Run..."
echo "   Project: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"
echo "   Region:  $REGION"
echo ""

# Build and deploy
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --set-env-vars "GROQ_API_KEY=${GROQ_API_KEY},GEMINI_API_KEY=${GEMINI_API_KEY}" \
    --min-instances 0 \
    --max-instances 2

echo ""
echo "✅ Deployment complete!"
echo "   Your app is live at:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
