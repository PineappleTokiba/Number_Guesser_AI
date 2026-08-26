const canvas = document.querySelector('#grid');
const ctx = canvas.getContext('2d');
let gesture = 'idle';
let panStart = null;
let panX = 0;
let panY = 0;
let panOrigin = null;
let currentStroke = null;
const strokes = [];
const activePointers = new Map();
let model = 'cnn';
const MODEL_CONFIG = {
  mlp: {
    description: 'Multilayer perceptron · raw pixels',
  },
  cnn: {
    description: 'Convolutional network · normalized input',
  },
  emnist: {
    description: 'EMNIST ByMerge · digits and letters',
  },
};

async function predict(modelName) {
  const modelPixels = createModelInput();
  if (!modelPixels) {
    throw new Error('Draw a character first');
  }

  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelName, pixels: modelPixels }),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || 'Prediction failed');
  }
  return result;
}

function clearCanvas() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#10100e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawStroke(context, stroke) {
  context.strokeStyle = '#f4f0e8';
  context.fillStyle = '#f4f0e8';
  context.lineWidth = stroke.width;
  context.lineCap = 'round';
  context.lineJoin = 'round';

  if (stroke.points.length === 1) {
    const point = stroke.points[0];
    context.beginPath();
    context.arc(point.x, point.y, stroke.width / 2, 0, Math.PI * 2);
    context.fill();
    return;
  }

  context.beginPath();
  context.moveTo(stroke.points[0].x, stroke.points[0].y);
  for (const point of stroke.points.slice(1)) {
    context.lineTo(point.x, point.y);
  }
  context.stroke();
}

function renderCanvas() {
  clearCanvas();
  const pixelRatio = window.devicePixelRatio || 1;
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, panX * pixelRatio, panY * pixelRatio);
  for (const stroke of strokes) drawStroke(ctx, stroke);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function pointerCenter() {
  const points = Array.from(activePointers.values());
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function updateDotPosition() {
  document.querySelector('.canvas-wrap').style.setProperty('--pan-x', `${panX}px`);
  document.querySelector('.canvas-wrap').style.setProperty('--pan-y', `${panY}px`);
}

function pixelIntensity(red, green, blue) {
  const brightness = (red + green + blue) / 3;
  return Math.max(0, Math.min(1, (brightness - 16) / (244 - 16)));
}

function createModelInput() {
  if (strokes.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const stroke of strokes) {
    const radius = stroke.width / 2;
    for (const point of stroke.points) {
      minX = Math.min(minX, point.x - radius);
      minY = Math.min(minY, point.y - radius);
      maxX = Math.max(maxX, point.x + radius);
      maxY = Math.max(maxY, point.y + radius);
    }
  }

  const cropWidth = maxX - minX;
  const cropHeight = maxY - minY;
  const scale = Math.min(20 / cropWidth, 20 / cropHeight);
  const resizedWidth = cropWidth * scale;
  const resizedHeight = cropHeight * scale;
  const destinationX = (28 - resizedWidth) / 2;
  const destinationY = (28 - resizedHeight) / 2;

  const modelCanvas = document.createElement('canvas');
  modelCanvas.width = 28;
  modelCanvas.height = 28;
  const modelContext = modelCanvas.getContext('2d');
  modelContext.fillStyle = '#10100e';
  modelContext.fillRect(0, 0, 28, 28);
  modelContext.imageSmoothingEnabled = true;
  modelContext.imageSmoothingQuality = 'high';
  modelContext.save();
  modelContext.translate(destinationX, destinationY);
  modelContext.scale(scale, scale);
  modelContext.translate(-minX, -minY);
  for (const stroke of strokes) drawStroke(modelContext, stroke);
  modelContext.restore();

  const normalized = modelContext.getImageData(0, 0, 28, 28);
  const pixels = new Array(28 * 28);
  for (let index = 0; index < pixels.length; index++) {
    const offset = index * 4;
    pixels[index] = pixelIntensity(
      normalized.data[offset],
      normalized.data[offset + 1],
      normalized.data[offset + 2],
    );
  }
  return pixels;
}

function resizeCanvas() {
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * pixelRatio);
  canvas.height = Math.round(canvas.clientHeight * pixelRatio);
  renderCanvas();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function worldPoint(screenPoint) {
  return { x: screenPoint.x - panX, y: screenPoint.y - panY };
}

canvas.addEventListener('pointerdown', event => {
  const point = canvasPoint(event);
  activePointers.set(event.pointerId, point);
  canvas.setPointerCapture(event.pointerId);

  if (activePointers.size === 1) {
    gesture = 'draw';
    const brush = Number(document.querySelector('#brush').value);
    currentStroke = {
      width: 8 + brush * 6,
      points: [worldPoint(point)],
    };
    strokes.push(currentStroke);
    renderCanvas();
  } else if (activePointers.size === 2) {
    // Remove the first touch's provisional dot before starting a pan gesture.
    if (currentStroke && strokes.at(-1) === currentStroke) strokes.pop();
    currentStroke = null;
    gesture = 'pan';
    panStart = pointerCenter();
    panOrigin = { x: panX, y: panY };
    renderCanvas();
  }
});

canvas.addEventListener('pointermove', event => {
  if (!activePointers.has(event.pointerId)) return;
  activePointers.set(event.pointerId, canvasPoint(event));

  if (gesture === 'pan' && activePointers.size >= 2) {
    const center = pointerCenter();
    const offsetX = center.x - panStart.x;
    const offsetY = center.y - panStart.y;
    panX = panOrigin.x + offsetX;
    panY = panOrigin.y + offsetY;
    renderCanvas();
    updateDotPosition();
  } else if (gesture === 'draw' && activePointers.size === 1) {
    currentStroke.points.push(worldPoint(canvasPoint(event)));
    renderCanvas();
  }
});

function finishPointer(event) {
  activePointers.delete(event.pointerId);
  currentStroke = null;

  if (activePointers.size === 0) {
    gesture = 'idle';
    panStart = null;
  } else if (gesture === 'pan') {
    // Wait for every finger to lift instead of accidentally drawing with the last one.
    gesture = 'wait';
  }
}

canvas.addEventListener('pointerup', finishPointer);
canvas.addEventListener('pointercancel', finishPointer);

canvas.addEventListener('wheel', event => {
  event.preventDefault();
  panX -= event.deltaX;
  panY -= event.deltaY;
  renderCanvas();
  updateDotPosition();
}, { passive: false });
window.addEventListener('resize', resizeCanvas);

document.querySelector('#clear').addEventListener('click', () => { strokes.length=0; panX=0; panY=0; renderCanvas(); updateDotPosition(); document.querySelector('#prediction').textContent='—'; document.querySelector('#confidence').textContent='Draw a character to begin'; document.querySelector('#bars').innerHTML=''; });
document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => {
  model=button.dataset.model; document.querySelectorAll('[data-model]').forEach(b=>b.classList.toggle('active',b===button));
  document.querySelector('#description').textContent = MODEL_CONFIG[model].description;
  document.querySelector('#prediction').textContent='—';
  document.querySelector('#bars').innerHTML='';
}));

document.querySelector('#predict').addEventListener('click', async () => {
  const button=document.querySelector('#predict'); button.disabled=true; button.firstChild.textContent='Thinking… ';
  try {
    const result=await predict(model);
    document.querySelector('#prediction').textContent=result.prediction;
    document.querySelector('#confidence').textContent=`Top three probabilities · ${result.model.toUpperCase()}`;
    document.querySelector('#bars').innerHTML=result.topThree.map(({ probability, label }, rank)=>`<div class="bar-row ${rank===0?'winner':''}"><span>${label}</span><div class="track"><div class="fill" style="width:${probability*100}%"></div></div><span>${(probability*100).toFixed(1)}%</span></div>`).join('');
  } catch(error) { document.querySelector('#confidence').textContent=`Error: ${error.message}`; }
  finally { button.disabled=false; button.firstChild.textContent='Recognize character '; }
});
resizeCanvas();
