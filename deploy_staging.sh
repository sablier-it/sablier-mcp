#!/bin/bash
# Cloud Run Deployment for Sablier MCP Server - STAGING
# Deploys to sablier-mcp-staging, pointing at staging backend API.
# Usage: ./deploy_staging.sh [--rebuild]

set -e

echo "Deploying Sablier MCP Server to Cloud Run (STAGING)..."

if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found."
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID="sablier-ai"
fi

SERVICE_NAME="sablier-mcp-staging"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/sablier-mcp"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)' 2>/dev/null || echo '215397666394')
SERVICE_URL="https://${SERVICE_NAME}-${PROJECT_NUMBER}.${REGION}.run.app"

# Staging points at the staging backend API
STAGING_API_URL="https://sablier-api-staging-${PROJECT_NUMBER}.${REGION}.run.app/api/v1"

REBUILD=false
for arg in "$@"; do
    case $arg in
        --rebuild) REBUILD=true ;;
    esac
done

if [[ "$REBUILD" == true ]]; then
    echo "Building Docker image..."
    gcloud builds submit \
        --tag "${IMAGE_NAME}:latest" \
        --project=${PROJECT_ID} \
        --timeout=10m \
        .
    echo "Image built successfully"
else
    echo "Using existing image..."
fi

echo "Deploying to Cloud Run..."
echo "  Service: ${SERVICE_NAME} (STAGING)"
echo "  URL: ${SERVICE_URL}"
echo "  Backend API: ${STAGING_API_URL}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 600 \
    --max-instances 2 \
    --min-instances 0 \
    --concurrency 40 \
    --set-env-vars "MCP_TRANSPORT=streamable-http,MCP_ISSUER_URL=${SERVICE_URL},SABLIER_API_URL=${STAGING_API_URL}" \
    --set-secrets 'MCP_TOKEN_SECRET=mcp-token-secret:latest' \
    --quiet

DEPLOYED_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format="value(status.url)" 2>/dev/null)
if [ -n "$DEPLOYED_URL" ]; then
    echo ""
    echo "STAGING deployed! ${DEPLOYED_URL}/mcp/"
    echo ""
    echo "Staging Configuration:"
    echo "  - Backend: ${STAGING_API_URL}"
    echo "  - Auto-scaling: 0-2 instances"
    echo "  - Same token secret as production (shared logins)"
    echo ""
    echo "To test with Claude Desktop, add to claude_desktop_config.json:"
    echo "  \"sablier-staging\": {"
    echo "    \"type\": \"url\","
    echo "    \"url\": \"${DEPLOYED_URL}/mcp/\""
    echo "  }"
else
    echo "ERROR: Deployment failed."
    exit 1
fi
