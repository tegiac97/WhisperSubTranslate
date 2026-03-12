const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { execSync } = require('child_process');
const axios = require('axios');
const ffmpegStatic = require('ffmpeg-static');
const EnhancedSubtitleTranslator = require('./translator-enhanced');

const app = express();
const translator = new EnhancedSubtitleTranslator();

const PORT = Number(process.env.PORT || 8080);
const HOST = process.env.HOST || '0.0.0.0';
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'base';
const DEFAULT_LANGUAGE = process.env.DEFAULT_LANGUAGE || 'auto';
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 1024);

const ROOT_DIR = __dirname;
const MODELS_DIR = path.join(ROOT_DIR, '_models');
const UPLOAD_DIR = path.join(ROOT_DIR, 'uploads');
const OUTPUT_DIR = path.join(ROOT_DIR, 'outputs');
const TMP_DIR = path.join(ROOT_DIR, 'tmp');

const MODEL_URL_MAP = {
  tiny: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
  base: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
  small: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
  medium: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
  large: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
  'large-v2': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin',
  'large-v3': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin',
  'large-v3-turbo': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin'
};

const MODEL_FILE_MAP = {
  tiny: 'ggml-tiny.bin',
  base: 'ggml-base.bin',
  small: 'ggml-small.bin',
  medium: 'ggml-medium.bin',
  large: 'ggml-large.bin',
  'large-v2': 'ggml-large-v2.bin',
  'large-v3': 'ggml-large-v3.bin',
  'large-v3-turbo': 'ggml-large-v3-turbo.bin'
};

const upload = multer({
  dest: UPLOAD_DIR,
  limits: {
    fileSize: MAX_UPLOAD_MB * 1024 * 1024
  }
});

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use('/outputs', express.static(OUTPUT_DIR));

ensureDirectories();

// ===== GPU detection =====
const CUDA12_MIN_COMPUTE = 5.0;
let _gpuInfoCache = null;

function getGpuInfo() {
  if (_gpuInfoCache !== null) return _gpuInfoCache;
  try {
    const raw = execSync(
      'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader',
      { encoding: 'utf8', timeout: 3000, stdio: ['ignore', 'pipe', 'ignore'] }
    ).trim();
    if (!raw) { _gpuInfoCache = { available: false }; return _gpuInfoCache; }
    const parts = raw.split('\n')[0].split(',').map((s) => s.trim());
    const name = parts[0] || 'Unknown GPU';
    const computeCap = parseFloat(parts[1]) || 0;
    _gpuInfoCache = {
      available: true,
      name,
      computeCap,
      cudaCompatible: computeCap >= CUDA12_MIN_COMPUTE
    };
    console.log(`[GPU] ${name}, Compute ${computeCap}, CUDA12 ok: ${_gpuInfoCache.cudaCompatible}`);
  } catch (_err) {
    try {
      execSync('nvidia-smi -L', { stdio: 'ignore', timeout: 2000 });
      _gpuInfoCache = { available: true, name: 'Unknown NVIDIA GPU', computeCap: 0, cudaCompatible: false };
    } catch (_err2) {
      _gpuInfoCache = { available: false };
    }
  }
  return _gpuInfoCache;
}

function resolveDevice(requested) {
  const req = String(requested || 'auto').toLowerCase();
  const gpu = getGpuInfo();
  if (req === 'auto') return (gpu.available && gpu.cudaCompatible) ? 'cuda' : 'cpu';
  if (req === 'cuda') return (gpu.available && gpu.cudaCompatible) ? 'cuda' : 'cpu';
  return 'cpu';
}

// ===========================

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'WhisperSubTranslate API',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/gpu', (_req, res) => {
  const info = getGpuInfo();
  res.json({ ok: true, gpu: info });
});

app.get('/api/models', (_req, res) => {
  const models = Object.keys(MODEL_URL_MAP).map((name) => {
    const modelPath = path.join(MODELS_DIR, MODEL_FILE_MAP[name]);
    return {
      name,
      installed: fs.existsSync(modelPath)
    };
  });

  res.json({ ok: true, models });
});

app.post('/api/models/download', async (req, res) => {
  const model = String(req.body?.model || DEFAULT_MODEL);
  try {
    const modelPath = await ensureModel(model);
    res.json({ ok: true, model, modelPath });
  } catch (error) {
    res.status(400).json({ ok: false, error: error.message });
  }
});

app.post('/api/translate/text', async (req, res) => {
  const text = String(req.body?.text || '').trim();
  const method = String(req.body?.method || 'mymemory');
  const targetLang = String(req.body?.targetLang || 'ko');

  if (!text) {
    return res.status(400).json({ ok: false, error: 'text is required' });
  }

  try {
    const translatedText = await translator.translateAuto(text, method, targetLang);
    return res.json({ ok: true, translatedText, method, targetLang });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.get('/api/config/keys', (_req, res) => {
  try {
    const keys = translator.loadApiKeys();
    return res.json({
      ok: true,
      configured: {
        deepl: Boolean(keys.deepl),
        openai: Boolean(keys.openai),
        gemini: Boolean(keys.gemini),
        deepseek: Boolean(keys.deepseek)
      },
      preferredService: keys.preferredService || 'mymemory'
    });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/config/keys', async (req, res) => {
  try {
    const keys = req.body || {};
    const saveOk = translator.saveApiKeys(keys);
    if (!saveOk) {
      return res.status(500).json({ ok: false, error: 'failed to save api keys' });
    }

    if (typeof translator.loadApiKeys === 'function') {
      translator.apiKeys = translator.loadApiKeys();
    }

    return res.json({ ok: true });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/translate/srt', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, error: 'multipart field "file" is required' });
  }

  const method = String(req.body?.method || 'mymemory');
  const targetLang = String(req.body?.targetLang || 'ko');
  const sourceLang = String(req.body?.sourceLang || 'auto');

  const inputPath = req.file.path;
  const outputPath = path.join(OUTPUT_DIR, `${safeBasename(req.file.originalname, '.srt')}_${targetLang}.srt`);

  try {
    await translator.translateSRTFile(inputPath, outputPath, method, targetLang, null, sourceLang);
    safeUnlink(inputPath);

    return res.json({
      ok: true,
      outputFile: path.basename(outputPath),
      downloadUrl: `${BASE_URL}/outputs/${encodeURIComponent(path.basename(outputPath))}`
    });
  } catch (error) {
    safeUnlink(inputPath);
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/extract', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, error: 'multipart field "file" is required' });
  }

  const model = String(req.body?.model || DEFAULT_MODEL);
  const language = String(req.body?.language || DEFAULT_LANGUAGE);
  const device = resolveDevice(req.body?.device);
  const cliPath = resolveWhisperCliPath();

  if (!cliPath) {
    safeUnlink(req.file.path);
    return res.status(500).json({
      ok: false,
      error: 'whisper-cli not found. Set WHISPER_CLI_PATH or install whisper.cpp and add whisper-cli to PATH.'
    });
  }

  try {
    const modelPath = await ensureModel(model);
    const wavPath = await convertToWav(req.file.path);
    const outputBase = path.join(OUTPUT_DIR, safeBasename(req.file.originalname));
    const srtPath = `${outputBase}.srt`;

    const args = ['-m', modelPath, '-f', wavPath, '-osrt', '-of', outputBase];
    args.push('-l', (language && language !== 'auto') ? language : 'auto');
    if (device !== 'cuda') args.push('-ng'); // disable GPU when not using CUDA
    console.log(`[Extract] device=${device}, args: ${args.join(' ')}`);

    await runCommand(cliPath, args);

    safeUnlink(req.file.path);
    safeUnlink(wavPath);

    return res.json({
      ok: true,
      device,
      outputFile: path.basename(srtPath),
      downloadUrl: `${BASE_URL}/outputs/${encodeURIComponent(path.basename(srtPath))}`
    });
  } catch (error) {
    safeUnlink(req.file.path);
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/extract-and-translate', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, error: 'multipart field "file" is required' });
  }

  const model = String(req.body?.model || DEFAULT_MODEL);
  const language = String(req.body?.language || DEFAULT_LANGUAGE);
  const method = String(req.body?.method || 'mymemory');
  const targetLang = String(req.body?.targetLang || 'ko');
  const sourceLang = String(req.body?.sourceLang || 'auto');
  const device = resolveDevice(req.body?.device);

  const cliPath = resolveWhisperCliPath();
  if (!cliPath) {
    safeUnlink(req.file.path);
    return res.status(500).json({
      ok: false,
      error: 'whisper-cli not found. Set WHISPER_CLI_PATH or install whisper.cpp and add whisper-cli to PATH.'
    });
  }

  try {
    const modelPath = await ensureModel(model);
    const wavPath = await convertToWav(req.file.path);
    const base = path.join(OUTPUT_DIR, safeBasename(req.file.originalname));
    const extractedSrt = `${base}.srt`;
    const translatedSrt = `${base}_${targetLang}.srt`;

    const args = ['-m', modelPath, '-f', wavPath, '-osrt', '-of', base];
    args.push('-l', language === 'auto' ? 'auto' : language);
    if (device !== 'cuda') args.push('-ng'); // disable GPU when not using CUDA
    console.log(`[Extract+Translate] device=${device}, args: ${args.join(' ')}`);

    await runCommand(cliPath, args);
    await translator.translateSRTFile(extractedSrt, translatedSrt, method, targetLang, null, sourceLang);

    safeUnlink(req.file.path);
    safeUnlink(wavPath);

    return res.json({
      ok: true,
      device,
      extracted: {
        outputFile: path.basename(extractedSrt),
        downloadUrl: `${BASE_URL}/outputs/${encodeURIComponent(path.basename(extractedSrt))}`
      },
      translated: {
        outputFile: path.basename(translatedSrt),
        downloadUrl: `${BASE_URL}/outputs/${encodeURIComponent(path.basename(translatedSrt))}`
      }
    });
  } catch (error) {
    safeUnlink(req.file.path);
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.use((error, _req, res, _next) => {
  if (error && error.name === 'MulterError') {
    return res.status(400).json({ ok: false, error: error.message });
  }
  return res.status(500).json({ ok: false, error: error?.message || 'Internal server error' });
});

app.listen(PORT, HOST, () => {
  console.log(`[API] Listening on ${HOST}:${PORT}`);
  console.log(`[API] Health: ${BASE_URL}/health`);
});

function ensureDirectories() {
  for (const dir of [MODELS_DIR, UPLOAD_DIR, OUTPUT_DIR, TMP_DIR]) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }
}

function safeBasename(filename, fallbackExt = '') {
  const ext = path.extname(filename || '');
  const base = path.basename(filename || `file${fallbackExt}`, ext || fallbackExt);
  return base.replace(/[^a-zA-Z0-9._-]/g, '_');
}

async function ensureModel(modelName) {
  const normalized = String(modelName || '').trim();
  const modelFile = MODEL_FILE_MAP[normalized];
  const modelUrl = MODEL_URL_MAP[normalized];

  if (!modelFile || !modelUrl) {
    throw new Error(`Unsupported model: ${modelName}`);
  }

  const modelPath = path.join(MODELS_DIR, modelFile);
  if (fs.existsSync(modelPath)) {
    return modelPath;
  }

  console.log(`[Model] Downloading ${normalized} from Hugging Face...`);
  const response = await axios({
    method: 'GET',
    url: modelUrl,
    responseType: 'stream'
  });

  await new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(modelPath);
    response.data.pipe(writer);
    writer.on('finish', resolve);
    writer.on('error', reject);
  });

  return modelPath;
}

function resolveWhisperCliPath() {
  const envPath = process.env.WHISPER_CLI_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return envPath;
  }

  const localCandidates = [
    path.join(ROOT_DIR, 'whisper-cpp', 'build', 'bin', 'whisper-cli'),
    path.join(ROOT_DIR, 'whisper-cpp', 'whisper-cli')
  ];

  for (const candidate of localCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  try {
    execSync('command -v whisper-cli', { stdio: 'ignore' });
    return 'whisper-cli';
  } catch (_error) {
    return null;
  }
}

async function convertToWav(inputPath) {
  const wavPath = path.join(TMP_DIR, `${safeBasename(path.basename(inputPath, path.extname(inputPath)))}_${Date.now()}.wav`);
  const ffmpegPath = ffmpegStatic || 'ffmpeg';

  const args = [
    '-y',
    '-i', inputPath,
    '-ac', '1',
    '-ar', '16000',
    wavPath
  ];

  await runCommand(ffmpegPath, args);
  return wavPath;
}

async function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stderr = '';

    child.stdout.on('data', (chunk) => {
      process.stdout.write(chunk.toString());
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(text);
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed (${command}) code=${code}: ${stderr.slice(-5000)}`));
      }
    });
  });
}

function safeUnlink(filePath) {
  try {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (_error) {
    // ignore
  }
}
