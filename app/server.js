const http = require('http');
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

const ROOT = __dirname;
const STATIC = path.join(ROOT, 'static');
const PORT = Number(process.env.PORT) || 3001;
const MAX_BODY_SIZE = 20_000;

const modelConfigs = {
  mlp: {
    path: path.join(STATIC, 'models', 'mnist_mlp_model.onnx'),
    labels: '0123456789'.split(''),
  },
  cnn: {
    path: path.join(STATIC, 'models', 'mnist_cnn_model.onnx'),
    labels: '0123456789'.split(''),
    normalize: true,
  },
  emnist: {
    path: path.join(STATIC, 'models', 'emnist_bymerge_model.onnx'),
    labels: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'.split(''),
    transpose: true,
  },
};

const sessions = {};

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

function serveStatic(request, response) {
  const requestPath = request.url === '/' ? '/index.html' : new URL(request.url, 'http://localhost').pathname;
  const filePath = path.resolve(STATIC, `.${requestPath}`);
  if (!filePath.startsWith(`${STATIC}${path.sep}`)) {
    response.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(filePath, (error, data) => {
    if (error) {
      response.writeHead(error.code === 'ENOENT' ? 404 : 500).end('Not found');
      return;
    }
    response.writeHead(200, {
      'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    response.end(data);
  });
}

function sendJson(response, status, value) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(value));
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let body = '';

    request.setEncoding('utf8');
    request.on('data', chunk => {
      body += chunk;
      if (body.length > MAX_BODY_SIZE) {
        reject(new Error('Request body is too large'));
        request.destroy();
      }
    });
    request.on('end', () => {
      try {
        resolve(JSON.parse(body));
      } catch {
        reject(new Error('Request body must be valid JSON'));
      }
    });
    request.on('error', reject);
  });
}

function softmax(values) {
  const max = Math.max(...values);
  const exponents = values.map(value => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map(value => value / total);
}

async function predict(modelName, pixels) {
  const config = modelConfigs[modelName];
  if (!config) throw new Error('Unknown model');
  if (!Array.isArray(pixels) || pixels.length !== 28 * 28) {
    throw new Error('pixels must contain exactly 784 values');
  }
  if (!pixels.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new Error('Every pixel must be a number between 0 and 1');
  }

  if (!sessions[modelName]) {
    sessions[modelName] = ort.InferenceSession.create(config.path);
  }
  const session = await sessions[modelName];
  const inputPixels = new Float32Array(28 * 28);

  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const sourceIndex = config.transpose ? x * 28 + y : y * 28 + x;
    const rawValue = pixels[sourceIndex];
    inputPixels[y * 28 + x] = config.normalize
      ? (rawValue - 0.1307) / 0.3081
      : rawValue;
  }

  const input = new ort.Tensor('float32', inputPixels, [1, 1, 28, 28]);
  const output = await session.run({ image: input });
  const probabilities = softmax(Array.from(output.logits.data));
  const topThree = probabilities
    .map((probability, index) => ({ label: config.labels[index], probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);

  return { model: modelName, prediction: topThree[0].label, topThree };
}

async function handlePrediction(request, response) {
  try {
    const { model, pixels } = await readJson(request);
    sendJson(response, 200, await predict(model, pixels));
  } catch (error) {
    const clientError = error.message === 'Unknown model'
      || error.message.startsWith('pixels')
      || error.message.startsWith('Every pixel')
      || error.message.startsWith('Request body');
    sendJson(response, clientError ? 400 : 500, { error: error.message });
  }
}

function main() {
  const server = http.createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (request.method === 'POST' && pathname === '/api/predict') {
      await handlePrediction(request, response);
    }
    else if (request.method === 'GET' || request.method === 'HEAD') serveStatic(request, response);
    else response.writeHead(405).end('Method not allowed');
  });

  server.listen(PORT, '127.0.0.1', () => {
    console.log(`Character Lab running at http://localhost:${PORT}`);
  });
}

main();
