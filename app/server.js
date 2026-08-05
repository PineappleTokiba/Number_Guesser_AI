const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const STATIC = path.join(ROOT, 'static');
const PORT = Number(process.env.PORT) || 3001;

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
    response.writeHead(200, { 'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream' });
    response.end(data);
  });
}

function main() {
  const server = http.createServer(async (request, response) => {
    if (request.method === 'GET' || request.method === 'HEAD') serveStatic(request, response);
    else response.writeHead(405).end('Method not allowed');
  });

  server.listen(PORT, '127.0.0.1', () => {
    console.log(`Digit Lab running at http://localhost:3001`);
  });
}

main();
