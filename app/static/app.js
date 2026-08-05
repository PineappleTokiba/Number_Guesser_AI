const canvas = document.querySelector('#grid');
const ctx = canvas.getContext('2d');
const pixels = Array.from({ length: 28 }, () => Array(28).fill(0));
let drawing = false;
let model = 'cnn';
const sessions = {};

// Use an absolute URL here. ONNX Runtime loads its ES module dynamically, and a
// relative prefix can otherwise be resolved from the module's own directory,
// resulting in `/vendor/onnxruntime-web/vendor/onnxruntime-web/...`.
ort.env.wasm.wasmPaths = `${window.location.origin}/vendor/onnxruntime-web/`;

function softmax(values) {
  const max = Math.max(...values);
  const exponents = values.map(value => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map(value => value / total);
}

async function predict(modelName) {
  if (!sessions[modelName]) {
    sessions[modelName] = await ort.InferenceSession.create(`./models/mnist_${modelName}_model.onnx`);
  }

  const inputPixels = new Float32Array(28 * 28);
  let index = 0;
  for (const row of pixels) for (const rawValue of row) {
    inputPixels[index++] = modelName === 'cnn' ? (rawValue - 0.1307) / 0.3081 : rawValue;
  }

  const input = new ort.Tensor('float32', inputPixels, [1, 1, 28, 28]);
  const output = await sessions[modelName].run({ image: input });
  const probabilities = softmax(Array.from(output.logits.data));
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  return { prediction, confidence: probabilities[prediction], probabilities, model: modelName };
}

function render() {
  const size = canvas.width / 28;
  ctx.fillStyle = '#10100e'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    if (pixels[y][x]) { ctx.fillStyle = `rgba(244,240,232,${pixels[y][x]})`; ctx.fillRect(x*size, y*size, size, size); }
  }
  ctx.strokeStyle = 'rgba(244,240,232,.065)'; ctx.lineWidth = 1;
  for (let i=1;i<28;i++) { ctx.beginPath(); ctx.moveTo(i*size,0); ctx.lineTo(i*size,canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0,i*size); ctx.lineTo(canvas.width,i*size); ctx.stroke(); }
}

function paint(event) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * 28);
  const y = Math.floor((event.clientY - rect.top) / rect.height * 28);
  const radius = Number(document.querySelector('#brush').value);
  for (let dy=-radius;dy<=radius;dy++) for(let dx=-radius;dx<=radius;dx++) {
    const px=x+dx, py=y+dy, distance=Math.hypot(dx,dy);
    if(px>=0&&px<28&&py>=0&&py<28&&distance<=radius+.2) pixels[py][px]=Math.max(pixels[py][px], Math.max(.2, 1-distance/(radius+1)));
  }
  render();
}
canvas.addEventListener('pointerdown', e => { drawing=true; canvas.setPointerCapture(e.pointerId); paint(e); });
canvas.addEventListener('pointermove', paint);
canvas.addEventListener('pointerup', () => drawing=false);
canvas.addEventListener('pointercancel', () => drawing=false);

document.querySelector('#clear').addEventListener('click', () => { pixels.forEach(row=>row.fill(0)); render(); document.querySelector('#prediction').textContent='—'; document.querySelector('#confidence').textContent='Draw a number to begin'; document.querySelector('#bars').innerHTML=''; });
document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => {
  model=button.dataset.model; document.querySelectorAll('[data-model]').forEach(b=>b.classList.toggle('active',b===button));
  document.querySelector('#description').textContent = model==='cnn' ? 'Convolutional network · normalized input' : 'Multilayer perceptron · raw pixels';
}));

document.querySelector('#predict').addEventListener('click', async () => {
  const button=document.querySelector('#predict'); button.disabled=true; button.firstChild.textContent='Thinking… ';
  try {
    const result=await predict(model);
    document.querySelector('#prediction').textContent=result.prediction;
    document.querySelector('#confidence').textContent=`${(result.confidence*100).toFixed(1)}% confidence · ${result.model.toUpperCase()}`;
    document.querySelector('#bars').innerHTML=result.probabilities.map((p,i)=>`<div class="bar-row ${i===result.prediction?'winner':''}"><span>${i}</span><div class="track"><div class="fill" style="width:${p*100}%"></div></div><span>${(p*100).toFixed(1)}%</span></div>`).join('');
  } catch(error) { document.querySelector('#confidence').textContent=`Error: ${error.message}`; }
  finally { button.disabled=false; button.firstChild.textContent='Recognize digit '; }
});
render();
