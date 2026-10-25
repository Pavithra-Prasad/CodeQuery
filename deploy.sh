#!/bin/bash
# CodeQuery AI — Hugging Face Spaces Deployment Script
# Usage: ./deploy.sh <hf-username> [space-name]
#
# Before the first deploy:
#   1. Create a Docker Space at https://huggingface.co/new-space
#   2. Add GROQ_API_KEY / GEMINI_API_KEY as secrets in the Space settings
#   3. Create a Write token at https://huggingface.co/settings/tokens

set -e

# Configuration
HF_USER="$1"
SPACE_NAME="${2:-codequery-ai}"

if [ -z "$HF_USER" ]; then
    echo "❌ Usage: ./deploy.sh <hf-username> [space-name]"
    exit 1
fi

SPACE_REPO="https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"
APP_URL="https://$(echo "$HF_USER-$SPACE_NAME" | tr '[:upper:]' '[:lower:]' | tr '_.' '--').hf.space"

# Only committed code is pushed, so refuse to deploy with pending changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ You have uncommitted changes. Commit them first, then re-run."
    exit 1
fi

# Point the 'hf' remote at the Space
if git remote get-url hf >/dev/null 2>&1; then
    git remote set-url hf "$SPACE_REPO"
else
    git remote add hf "$SPACE_REPO"
fi

echo "🚀 Deploying CodeQuery AI to Hugging Face Spaces..."
echo "   Space: $SPACE_REPO"
echo ""
echo "   If git asks for credentials: username = $HF_USER, password = your HF access token"
echo ""

# The Space mirrors this repo, so overwrite the starter files HF created with it
git push --force hf HEAD:main

echo ""
echo "✅ Code pushed! The Space is building now (about 5-10 minutes the first time)."
echo "   Build logs: $SPACE_REPO"
echo "   Your app:   $APP_URL"
