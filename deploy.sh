#!/bin/bash
# Cloud Run Deployment for Sablier MCP Server - PRODUCTION
# Usage: ./deploy.sh

set -e

echo "Deploying Sablier MCP Server to Cloud Run (PRODUCTION)..."

if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found."
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID="sablier-ai"
fi

SERVICE_NAME="sablier-mcp"
REGION="us-central1"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)' 2>/dev/null || echo '215397666394')
SERVICE_URL="https://${SERVICE_NAME}-${PROJECT_NUMBER}.${REGION}.run.app"

echo "Deploying to Cloud Run..."
echo "  Service: ${SERVICE_NAME}"
echo "  URL: ${SERVICE_URL}"

gcloud run deploy ${SERVICE_NAME} \
    --source . \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 600 \
    --max-instances 4 \
    --min-instances 0 \
    --concurrency 40 \
    --set-env-vars "MCP_TRANSPORT=streamable-http,MCP_ISSUER_URL=${SERVICE_URL},SABLIER_API_URL=https://sablier-api-${PROJECT_NUMBER}.${REGION}.run.app/api/v1" \
    --set-secrets 'MCP_TOKEN_SECRET=mcp-token-secret:latest' \
    --quiet

DEPLOYED_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format="value(status.url)" 2>/dev/null)
if [ -n "$DEPLOYED_URL" ]; then
    echo ""
    echo "PRODUCTION deployed! ${DEPLOYED_URL}/mcp/"
else
    echo "ERROR: Deployment failed."
    exit 1
fi
