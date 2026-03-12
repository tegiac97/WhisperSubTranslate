# Deploy WhisperSubTranslate API on Ubuntu Cloud

This project is originally an Electron desktop app. The included `api-server.js` exposes a headless HTTP API for cloud usage.

## 1) Create Ubuntu VM

Use Ubuntu 22.04 or 24.04.

Minimum recommended:
- CPU-only: 4 vCPU, 8 GB RAM, 50+ GB disk
- GPU (faster): NVIDIA GPU with proper driver + CUDA runtime

## 2) Install system dependencies

```bash
sudo apt update
sudo apt install -y git curl ffmpeg build-essential unzip
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v
```

## 3) Clone and install

```bash
git clone <your-repo-url> WhisperSubTranslate
cd WhisperSubTranslate
npm install
```

## 4) Install whisper.cpp CLI

### Option A (recommended): build from source

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j
cd ..
```

Set env var so API uses that binary:

```bash
export WHISPER_CLI_PATH="$PWD/whisper.cpp/build/bin/whisper-cli"
```

### Option B: use system PATH

If `whisper-cli` is already in PATH, no env var is needed.

## 5) Configure environment

```bash
export HOST=0.0.0.0
export PORT=8080
export BASE_URL="http://YOUR_SERVER_PUBLIC_IP:8080"
export DEFAULT_MODEL=base
export DEFAULT_LANGUAGE=auto
```

Optional API keys for translator engines:
- DeepL
- OpenAI
- Gemini

You can set them through API endpoint `/api/config/keys`.

## 6) Start API server

```bash
npm run start:api
```

Check health:

```bash
curl http://127.0.0.1:8080/health
```

Open firewall/security group for TCP `8080`.

## 7) Run with PM2 (production)

```bash
sudo npm i -g pm2
pm2 start "npm run start:api" --name whispersub-api
pm2 save
pm2 startup
```

## API endpoints

Base URL: `http://YOUR_SERVER_PUBLIC_IP:8080`

### Health check

```bash
curl -s http://YOUR_SERVER_PUBLIC_IP:8080/health
```

### List model status

```bash
curl -s http://YOUR_SERVER_PUBLIC_IP:8080/api/models
```

### Download model

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"model":"base"}'
```

### Translate plain text

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/translate/text \
  -H "Content-Type: application/json" \
  -d '{
    "text":"Hello world",
    "method":"mymemory",
    "targetLang":"ko"
  }'
```

### Save API keys (optional)

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/config/keys \
  -H "Content-Type: application/json" \
  -d '{
    "deepl": "YOUR_DEEPL_KEY",
    "openai": "YOUR_OPENAI_KEY",
    "gemini": "YOUR_GEMINI_KEY",
    "preferredService": "gemini"
  }'
```

### Translate SRT file

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/translate/srt \
  -F "file=@/path/to/input.srt" \
  -F "method=mymemory" \
  -F "targetLang=ja" \
  -F "sourceLang=auto"
```

### Extract subtitles from video/audio

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/extract \
  -F "file=@/path/to/video.mp4" \
  -F "model=base" \
  -F "language=auto"
```

### Extract and translate in one call

```bash
curl -s -X POST http://YOUR_SERVER_PUBLIC_IP:8080/api/extract-and-translate \
  -F "file=@/path/to/video.mp4" \
  -F "model=base" \
  -F "language=auto" \
  -F "method=mymemory" \
  -F "targetLang=ko" \
  -F "sourceLang=auto"
```

Returned JSON includes `downloadUrl`, for example:

```json
{
  "ok": true,
  "outputFile": "video_ko.srt",
  "downloadUrl": "http://YOUR_SERVER_PUBLIC_IP:8080/outputs/video_ko.srt"
}
```

## Notes

- First run may take time while downloading model files into `_models/`.
- Keep enough disk space for uploads, model files, and generated outputs.
- For internet-exposed deployment, put Nginx in front with HTTPS and request size limits.
