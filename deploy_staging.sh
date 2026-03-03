#!/bin/bash
# Cloud Run Deployment for Sablier MCP Server - STAGING
# Deploys to sablier-mcp-staging.
# Usage: ./deploy_staging.sh [--staging-backend]

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
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)' 2>/dev/null || echo '215397666394')
SERVICE_URL="https://${SERVICE_NAME}-${PROJECT_NUMBER}.${REGION}.run.app"

# Staging MCP defaults to staging backend.
# Override with --production-backend to use the production backend instead.
BACKEND_API_URL="https://sablier-api-staging-${PROJECT_NUMBER}.${REGION}.run.app/api/v1"

for arg in "$@"; do
    case $arg in
        --production-backend)
            BACKEND_API_URL="https://sablier-api-${PROJECT_NUMBER}.${REGION}.run.app/api/v1"
            echo "Using PRODUCTION backend API"
            ;;
    esac
done

echo "Deploying to Cloud Run..."
echo "  Service: ${SERVICE_NAME} (STAGING)"
echo "  URL: ${SERVICE_URL}"
echo "  Backend API: ${BACKEND_API_URL}"

gcloud run deploy ${SERVICE_NAME} \
    --source . \
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
    --set-env-vars "MCP_TRANSPORT=streamable-http,MCP_ISSUER_URL=${SERVICE_URL},SABLIER_API_URL=${BACKEND_API_URL}" \
    --set-secrets 'MCP_TOKEN_SECRET=mcp-token-secret:latest' \
    --quiet

DEPLOYED_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format="value(status.url)" 2>/dev/null)
if [ -n "$DEPLOYED_URL" ]; then
    echo ""
    echo "STAGING deployed! ${DEPLOYED_URL}/mcp/"
    echo ""
    echo "Staging Configuration:"
    echo "  - Backend: ${BACKEND_API_URL}"
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
