/**
 * ProctorAI Frontend — Real-time exam proctoring interface
 *
 * Streams webcam frames (1 fps) and microphone audio (every 4s)
 * over a WebSocket to the AI proctoring backend.
 */

'use strict';

// ── STATE ──────────────────────────────────────────────────────────────────
const state = {
  ws: null,
  sessionId: '',
  backendUrl: '',
  examRunning: false,
  connected: false,
  frameInterval: null,
  audioInterval: null,
  uptimeInterval: null,
  startTime: null,
  framesSent: 0,
  audioChunks: 0,
  eventsCount: 0,
  warningsCount: 0,
  sessionRisk: 0,
  frameRisk: 0,
  audioRisk: 0,
  mediaStream: null,
  audioContext: null,
  audioAnalyser: null,
  scriptProcessor: null,
  pcmSamples: [],
  mediaRecorder: null,
  audioChunksBuffer: [],
  reconnectAttempts: 0,
  maxReconnect: 5,
  reconnectTimer: null,
};

// ── DOM REFS ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const els = {
  webcam:          $('webcam'),
  canvas:          $('snapshot-canvas'),
  bboxCanvas:      $('bbox-canvas'),
  camOverlay:      $('cam-overlay'),
  camBadge:        $('cam-badge'),
  faceBadge:       $('face-badge'),
  micFill:         $('mic-fill'),
  micStatus:       $('mic-status'),
  wsStatus:        $('ws-status'),
  wsStatusText:    $('ws-status-text'),
  sessionDisplay:  $('session-id-display'),
  sessionInput:    $('session-id-input'),
  backendInput:    $('backend-url'),
  uptime:          $('uptime'),
  btnConnect:      $('btn-connect'),
  btnDisconnect:   $('btn-disconnect'),
  btnStart:        $('btn-start'),
  btnStop:         $('btn-stop'),
  framesSent:      $('frames-sent'),
  audioChunks:     $('audio-chunks'),
  eventsCount:     $('events-count'),
  warningsCount:   $('warnings-count'),
  riskGauge:       $('risk-gauge'),
  riskScore:       $('risk-score'),
  riskLabel:       $('risk-label'),
  sessionRiskBar:  $('session-risk-bar'),
  sessionRiskVal:  $('session-risk-val'),
  frameRiskBar:    $('frame-risk-bar'),
  frameRiskVal:    $('frame-risk-val'),
  audioRiskBar:    $('audio-risk-bar'),
  audioRiskVal:    $('audio-risk-val'),
  alertsList:      $('alerts-list'),
  transcriptBody:  $('transcript-body'),
  eventsList:      $('events-list'),
  debugLog:        $('debug-log'),
};

// ── CONSTANTS ─────────────────────────────────────────────────────────────
// Frame dimensions sent to the server. Bounding box coordinates returned by
// the server are in this pixel space and must be scaled to the display size.
const FRAME_W = 320;
const FRAME_H = 240;

// ── EVENT CHIP MAP ─────────────────────────────────────────────────────────
const chipMap = {
  'mobile_detected':      'chip-mobile',
  'cell phone':           'chip-mobile',
  'headset_detected':     'chip-headset',
  'multiple_persons':     'chip-multiple',
  'no_face':              'chip-noface',
  'looking_away':         'chip-looking',
  'bad_posture':          'chip-posture',
  'suspicious_movement':  'chip-movement',
  'suspicious_transcript':'chip-transcript',
  'keyword_detected':     'chip-transcript',
};

const chipTimeouts = {};

// ── LOGGING ───────────────────────────────────────────────────────────────
function log(msg, type = 'info') {
  const line = document.createElement('div');
  line.className = `debug-line ${type}`;
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${msg}`;
  els.debugLog.appendChild(line);
  els.debugLog.scrollTop = els.debugLog.scrollHeight;
  // Keep max 200 lines
  while (els.debugLog.children.length > 200) {
    els.debugLog.removeChild(els.debugLog.firstChild);
  }
}

// ── WS STATUS ─────────────────────────────────────────────────────────────
function setWsStatus(status) {
  const dot = els.wsStatus.querySelector('.dot');
  dot.className = 'dot';
  switch (status) {
    case 'connecting':
      dot.classList.add('dot-connecting');
      els.wsStatusText.textContent = 'Connecting...';
      break;
    case 'connected':
      dot.classList.add('dot-on');
      els.wsStatusText.textContent = 'Connected';
      break;
    case 'disconnected':
      dot.classList.add('dot-off');
      els.wsStatusText.textContent = 'Disconnected';
      break;
    case 'error':
      dot.classList.add('dot-error');
      els.wsStatusText.textContent = 'Error';
      break;
  }
}

// ── CONNECT ───────────────────────────────────────────────────────────────
function connect() {
  state.sessionId = els.sessionInput.value.trim() || 'exam-session-001';
  state.backendUrl = els.backendInput.value.trim() || 'http://localhost:8000';

  const wsUrl = state.backendUrl.replace(/^http/, 'ws') + `/ws/proctor/${state.sessionId}`;
  log(`Connecting to ${wsUrl}`, 'info');
  setWsStatus('connecting');

  try {
    state.ws = new WebSocket(wsUrl);
  } catch (e) {
    log(`WebSocket creation failed: ${e.message}`, 'error');
    setWsStatus('error');
    return;
  }

  state.ws.onopen = () => {
    state.connected = true;
    state.reconnectAttempts = 0;
    setWsStatus('connected');
    els.sessionDisplay.textContent = state.sessionId;
    els.btnConnect.disabled = true;
    els.btnDisconnect.disabled = false;
    els.btnStart.disabled = false;
    log(`Connected to session: ${state.sessionId}`, 'success');
  };

  state.ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleServerMessage(msg);
    } catch (e) {
      log(`Parse error: ${e.message}`, 'warn');
    }
  };

  state.ws.onclose = (evt) => {
    state.connected = false;
    setWsStatus('disconnected');
    els.btnConnect.disabled = false;
    els.btnDisconnect.disabled = true;
    log(`WebSocket closed (code ${evt.code})`, 'warn');

    if (state.examRunning && state.reconnectAttempts < state.maxReconnect) {
      state.reconnectAttempts++;
      const delay = Math.min(1000 * state.reconnectAttempts, 5000);
      log(`Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts})...`, 'warn');
      state.reconnectTimer = setTimeout(connect, delay);
    } else if (state.examRunning) {
      stopExam();
    }
  };

  state.ws.onerror = (err) => {
    log('WebSocket error', 'error');
    setWsStatus('error');
  };
}

// ── DISCONNECT ────────────────────────────────────────────────────────────
function disconnect() {
  if (state.examRunning) stopExam();
  clearTimeout(state.reconnectTimer);
  state.reconnectAttempts = state.maxReconnect; // prevent auto-reconnect
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  state.connected = false;
  setWsStatus('disconnected');
  els.btnConnect.disabled = false;
  els.btnDisconnect.disabled = true;
  els.btnStart.disabled = true;
  log('Disconnected', 'info');
}

// ── START EXAM ────────────────────────────────────────────────────────────
async function startExam() {
  if (!state.connected) { log('Not connected', 'warn'); return; }

  log('Starting exam session...', 'info');
  state.examRunning = true;
  state.startTime = Date.now();
  state.reconnectAttempts = 0;

  els.btnStart.disabled = true;
  els.btnStop.disabled = false;

  // Start uptime counter
  state.uptimeInterval = setInterval(updateUptime, 1000);

  // Start camera + mic
  await startMedia();

  // Send 1 frame per second
  state.frameInterval = setInterval(sendVideoFrame, 1000);

  // Send audio every 4 seconds
  state.audioInterval = setInterval(sendAudioChunk, 4000);

  log('Exam started — streaming video + audio', 'success');
}

// ── STOP EXAM ─────────────────────────────────────────────────────────────
function stopExam() {
  state.examRunning = false;
  clearInterval(state.frameInterval);
  clearInterval(state.audioInterval);
  clearInterval(state.uptimeInterval);

  stopMedia();

  els.btnStart.disabled = false;
  els.btnStop.disabled = true;
  log('Exam stopped', 'warn');
}

// ── MEDIA ─────────────────────────────────────────────────────────────────
async function startMedia() {
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: true,
    });

    els.webcam.srcObject = state.mediaStream;
    els.camOverlay.classList.add('hidden');
    els.camBadge.textContent = 'LIVE';
    els.camBadge.className = 'badge badge-green';
    els.micStatus.textContent = 'ON';

    // Mic level visualizer
    state.audioContext = new AudioContext();
    const source = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.audioAnalyser = state.audioContext.createAnalyser();
    state.audioAnalyser.fftSize = 256;
    source.connect(state.audioAnalyser);
    animateMicLevel();

    // Set up PCM capture via ScriptProcessor
    setupMediaRecorder();

    log('Camera and microphone started', 'success');
  } catch (e) {
    log(`Media error: ${e.message}`, 'error');
  }
}

function stopMedia() {
  if (state.scriptProcessor) {
    state.scriptProcessor.disconnect();
    state.scriptProcessor = null;
  }
  state.pcmSamples = [];
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(t => t.stop());
    state.mediaStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
    state.mediaRecorder.stop();
  }
  els.webcam.srcObject = null;
  els.camOverlay.classList.remove('hidden');
  els.camBadge.textContent = 'OFF';
  els.camBadge.className = 'badge';
  els.micStatus.textContent = 'OFF';
  els.micFill.style.width = '0%';
}

function setupMediaRecorder() {
  if (!state.mediaStream) return;
  // Use Web Audio API to capture raw PCM and encode as WAV
  // This avoids the WebM format issue with the STT service
  state.audioChunksBuffer = [];
  state.useWebAudioCapture = true;

  // Set up a ScriptProcessor to capture raw PCM samples
  if (state.audioContext) {
    const source = state.audioContext.createMediaStreamSource(state.mediaStream);
    const bufferSize = 4096;
    state.scriptProcessor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
    state.pcmSamples = [];

    state.scriptProcessor.onaudioprocess = (e) => {
      if (!state.examRunning) return;
      const channelData = e.inputBuffer.getChannelData(0);
      // Downsample to 16kHz if needed
      const inputSampleRate = state.audioContext.sampleRate;
      const ratio = inputSampleRate / 16000;
      const outputLength = Math.floor(channelData.length / ratio);
      const downsampled = new Float32Array(outputLength);
      for (let i = 0; i < outputLength; i++) {
        downsampled[i] = channelData[Math.floor(i * ratio)];
      }
      state.pcmSamples.push(downsampled);
    };

    source.connect(state.scriptProcessor);
    state.scriptProcessor.connect(state.audioContext.destination);
  }
}

function animateMicLevel() {
  if (!state.audioAnalyser) return;
  const data = new Uint8Array(state.audioAnalyser.frequencyBinCount);
  state.audioAnalyser.getByteFrequencyData(data);
  const avg = data.reduce((a, b) => a + b, 0) / data.length;
  const pct = Math.min(100, (avg / 128) * 100);
  els.micFill.style.width = pct + '%';
  if (state.examRunning) requestAnimationFrame(animateMicLevel);
}

// ── SEND VIDEO FRAME ──────────────────────────────────────────────────────
function sendVideoFrame() {
  if (!state.connected || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (!state.mediaStream) return;

  const video = els.webcam;
  const canvas = els.canvas;
  canvas.width = FRAME_W;
  canvas.height = FRAME_H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // JPEG at 70% quality — good balance of size vs detail
  const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];

  state.ws.send(JSON.stringify({ type: 'video', frame: b64 }));
  state.framesSent++;
  els.framesSent.textContent = state.framesSent;
}

// ── SEND AUDIO CHUNK ──────────────────────────────────────────────────────
async function sendAudioChunk() {
  if (!state.connected || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (!state.pcmSamples || state.pcmSamples.length === 0) return;

  // Grab buffered PCM samples and encode as WAV
  const samples = state.pcmSamples.splice(0);
  if (samples.length === 0) return;

  // Concatenate all Float32 chunks
  const totalLength = samples.reduce((acc, s) => acc + s.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of samples) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  // Encode as 16-bit PCM WAV at 16kHz
  const wavBytes = encodeWAV(merged, 16000);
  const b64 = arrayBufferToBase64(wavBytes);

  state.ws.send(JSON.stringify({ type: 'audio', audio: b64 }));
  state.audioChunks++;
  els.audioChunks.textContent = state.audioChunks;
  log(`Audio chunk sent (${Math.round(wavBytes.byteLength / 1024)}KB, ${merged.length} samples)`, 'info');
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  const byteRate = sampleRate * 2;
  const dataSize = samples.length * 2;

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);       // PCM chunk size
  view.setUint16(20, 1, true);        // PCM format
  view.setUint16(22, 1, true);        // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, 2, true);        // block align
  view.setUint16(34, 16, true);       // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert Float32 to Int16
  let sampleOffset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(sampleOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    sampleOffset += 2;
  }

  return buffer;
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// ── HANDLE SERVER MESSAGE ─────────────────────────────────────────────────
function handleServerMessage(msg) {
  // Heartbeat ping — respond with pong
  if (msg.type === 'ping') {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: 'pong' }));
    }
    return;
  }

  // Error message
  if (msg.type === 'error') {
    log(`Server error: ${msg.error}`, 'error');
    return;
  }

  // Audio analysis result — has native_text OR rule_flags (but NOT person_count)
  // Check audio FIRST because audio payload also has events[] from rule_flags
  if (msg.native_text !== undefined || msg.rule_flags !== undefined) {
    if (msg.person_count === undefined) {
      handleAudioResult(msg);
      return;
    }
  }

  // Video analysis result — has events[] AND person_count
  if (msg.events !== undefined && msg.person_count !== undefined) {
    handleVideoResult(msg);
    return;
  }

  // Generic event
  if (msg.event) {
    addEvent(msg.event, msg.risk_level || 'LOW', msg.timestamp);
    return;
  }

  log(`Unknown message: ${JSON.stringify(msg).substring(0, 80)}`, 'warn');
}

function handleVideoResult(msg) {
  const { events = [], person_count, risk_score, risk_level, detections = [] } = msg;

  state.frameRisk = risk_score || 0;
  state.sessionRisk += state.frameRisk;
  updateRiskDisplay(risk_score, risk_level);
  updateRiskBars();

  // Draw bounding boxes on the webcam overlay
  drawBoundingBoxes(detections, FRAME_W, FRAME_H);

  // Update face badge
  if (person_count === 0) {
    els.faceBadge.textContent = 'No Face';
    els.faceBadge.className = 'badge badge-red';
  } else if (person_count > 1) {
    els.faceBadge.textContent = `${person_count} Persons`;
    els.faceBadge.className = 'badge badge-yellow';
  } else {
    els.faceBadge.textContent = '1 Person';
    els.faceBadge.className = 'badge badge-green';
  }

  // Process events
  events.forEach(evt => {
    addEvent(evt, risk_level, msg.timestamp);
    flashChip(evt);
    if (risk_level === 'HIGH' || risk_level === 'MEDIUM') {
      addAlert(evt, risk_level, msg.timestamp);
    }
  });

  if (events.length > 0) {
    log(`Video: [${events.join(', ')}] risk=${risk_score} (${risk_level})`, risk_level === 'HIGH' ? 'error' : 'warn');
  }
}

function handleAudioResult(msg) {
  const { native_text, translated_text, rule_flags = [], risk_score, risk_level } = msg;

  state.audioRisk = risk_score || 0;
  state.sessionRisk += state.audioRisk;
  updateRiskDisplay(risk_score, risk_level);
  updateRiskBars();

  // Add transcript entry
  if (native_text) {
    addTranscript(native_text, translated_text, rule_flags);
  }

  // Process rule flags as events
  rule_flags.forEach(flag => {
    addEvent(flag, risk_level, msg.timestamp);
    flashChip(flag);
  });

  if (risk_level === 'HIGH' || risk_level === 'MEDIUM') {
    if (rule_flags.length > 0) {
      addAlert(`Speech: ${rule_flags.join(', ')}`, risk_level, msg.timestamp);
    }
  }

  log(`Audio: "${(native_text || '').substring(0, 50)}" risk=${risk_score} (${risk_level})`,
    risk_level === 'HIGH' ? 'error' : 'info');
}

// ── UI UPDATES ────────────────────────────────────────────────────────────
function updateRiskDisplay(score, level) {
  const s = score || 0;
  const l = (level || 'LOW').toLowerCase();

  els.riskScore.textContent = s;
  els.riskLabel.textContent = level || 'LOW';
  els.riskGauge.className = `risk-gauge ${l}`;

  // Color the score text
  if (l === 'high') {
    els.riskScore.style.color = 'var(--red)';
    els.riskLabel.style.color = 'var(--red)';
  } else if (l === 'medium') {
    els.riskScore.style.color = 'var(--yellow)';
    els.riskLabel.style.color = 'var(--yellow)';
  } else {
    els.riskScore.style.color = 'var(--green)';
    els.riskLabel.style.color = 'var(--green)';
  }
}

function updateRiskBars() {
  const maxScore = 20;

  const sessionPct = Math.min(100, (state.sessionRisk / maxScore) * 100);
  const framePct = Math.min(100, (state.frameRisk / maxScore) * 100);
  const audioPct = Math.min(100, (state.audioRisk / maxScore) * 100);

  setBar(els.sessionRiskBar, els.sessionRiskVal, sessionPct, state.sessionRisk);
  setBar(els.frameRiskBar, els.frameRiskVal, framePct, state.frameRisk);
  setBar(els.audioRiskBar, els.audioRiskVal, audioPct, state.audioRisk);
}

function setBar(barEl, valEl, pct, val) {
  barEl.style.width = pct + '%';
  valEl.textContent = val;
  barEl.className = 'progress-fill';
  if (pct >= 60) barEl.classList.add('high');
  else if (pct >= 30) barEl.classList.add('medium');
}

function addEvent(eventName, riskLevel, timestamp) {
  state.eventsCount++;
  els.eventsCount.textContent = state.eventsCount;

  const level = (riskLevel || 'LOW').toLowerCase();
  const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

  // Remove empty state
  const empty = els.eventsList.querySelector('.empty-state');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'event-item';
  item.innerHTML = `
    <div class="event-dot ${level}"></div>
    <div class="event-name">${formatEventName(eventName)}</div>
    <div class="event-time">${time}</div>
    <div class="event-level ${level}">${riskLevel || 'LOW'}</div>
  `;
  els.eventsList.insertBefore(item, els.eventsList.firstChild);

  // Keep max 100 events
  while (els.eventsList.children.length > 100) {
    els.eventsList.removeChild(els.eventsList.lastChild);
  }
}

function addAlert(eventName, riskLevel, timestamp) {
  state.warningsCount++;
  els.warningsCount.textContent = state.warningsCount;

  const level = (riskLevel || 'HIGH').toLowerCase();
  const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

  const empty = els.alertsList.querySelector('.empty-state');
  if (empty) empty.remove();

  const icons = { high: '🚨', medium: '⚠️', low: 'ℹ️' };
  const item = document.createElement('div');
  item.className = `alert-item ${level}`;
  item.innerHTML = `
    <div class="alert-icon">${icons[level] || '⚠️'}</div>
    <div class="alert-body">
      <div class="alert-event">${formatEventName(eventName)}</div>
      <div class="alert-time">${time}</div>
    </div>
  `;
  els.alertsList.insertBefore(item, els.alertsList.firstChild);

  while (els.alertsList.children.length > 50) {
    els.alertsList.removeChild(els.alertsList.lastChild);
  }
}

function addTranscript(nativeText, translatedText, flags) {
  const empty = els.transcriptBody.querySelector('.empty-state');
  if (empty) empty.remove();

  const entry = document.createElement('div');
  entry.className = 'transcript-entry';
  entry.innerHTML = `
    <div class="transcript-lang">🎙 Detected Speech ${flags.length > 0 ? '⚠️' : ''}</div>
    <div class="transcript-text">${escapeHtml(nativeText)}</div>
    ${translatedText && translatedText !== nativeText
      ? `<div class="transcript-translated">→ ${escapeHtml(translatedText)}</div>`
      : ''}
  `;
  els.transcriptBody.insertBefore(entry, els.transcriptBody.firstChild);

  while (els.transcriptBody.children.length > 20) {
    els.transcriptBody.removeChild(els.transcriptBody.lastChild);
  }
}

function flashChip(eventName) {
  const chipId = chipMap[eventName];
  if (!chipId) return;
  const chip = $(chipId);
  if (!chip) return;

  chip.classList.add('chip-active');
  chip.classList.remove('chip-off');

  clearTimeout(chipTimeouts[chipId]);
  chipTimeouts[chipId] = setTimeout(() => {
    chip.classList.remove('chip-active');
    chip.classList.add('chip-off');
  }, 3000);
}

// ── BOUNDING BOX DRAWING ──────────────────────────────────────────────────

// Color map for detected object classes
const CLASS_COLORS = {
  'person':     '#22c55e',   // green
  'cell phone': '#ef4444',   // red
  'mobile_detected': '#ef4444',
  'laptop':     '#f97316',   // orange
  'monitor':    '#f97316',
  'book':       '#eab308',   // yellow
  'default':    '#6366f1',   // purple
};

let _bboxFadeTimer = null;

function drawBoundingBoxes(detections, videoWidth, videoHeight) {
  const canvas = els.bboxCanvas;
  if (!canvas) return;

  // Match canvas size to the displayed video element
  const rect = els.webcam.getBoundingClientRect();
  canvas.width = rect.width || 320;
  canvas.height = rect.height || 240;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!detections || detections.length === 0) return;

  // The server receives the frame at the snapshot canvas resolution (320×240)
  // and Ultralytics returns bounding boxes scaled back to that input resolution.
  // Scale from the sent frame dimensions → displayed canvas size.
  const scaleX = canvas.width / videoWidth;
  const scaleY = canvas.height / videoHeight;

  detections.forEach(det => {
    const bb = det.bounding_box;
    if (!bb) return;

    const x = bb.x1 * scaleX;
    const y = bb.y1 * scaleY;
    const w = (bb.x2 - bb.x1) * scaleX;
    const h = (bb.y2 - bb.y1) * scaleY;

    const color = CLASS_COLORS[det.class_name] || CLASS_COLORS['default'];
    const conf = Math.round((det.confidence || 0) * 100);
    const label = `${formatEventName(det.class_name)} ${conf}%`;

    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Draw semi-transparent fill
    ctx.fillStyle = color + '22';
    ctx.fillRect(x, y, w, h);

    // Draw label background
    ctx.font = 'bold 11px monospace';
    const textW = ctx.measureText(label).width + 8;
    const textH = 18;
    const labelY = y > textH ? y - textH : y + h;

    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, textW, textH);

    // Draw label text
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 4, labelY + 13);
  });

  // Auto-clear boxes after 1.5s (they'll be redrawn on next frame)
  clearTimeout(_bboxFadeTimer);
  _bboxFadeTimer = setTimeout(() => {
    const c = els.bboxCanvas;
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }, 1500);
}

function updateUptime() {  if (!state.startTime) return;
  const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
  const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  els.uptime.textContent = `${h}:${m}:${s}`;
}

// ── HELPERS ───────────────────────────────────────────────────────────────
function formatEventName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── CLEAR BUTTONS ─────────────────────────────────────────────────────────
$('btn-clear-alerts').addEventListener('click', () => {
  els.alertsList.innerHTML = '<div class="empty-state">No alerts yet</div>';
  state.warningsCount = 0;
  els.warningsCount.textContent = '0';
});

$('btn-clear-events').addEventListener('click', () => {
  els.eventsList.innerHTML = '<div class="empty-state">No events yet</div>';
  state.eventsCount = 0;
  els.eventsCount.textContent = '0';
});

$('btn-clear-debug').addEventListener('click', () => {
  els.debugLog.innerHTML = '';
});

// ── BUTTON WIRING ─────────────────────────────────────────────────────────
els.btnConnect.addEventListener('click', connect);
els.btnDisconnect.addEventListener('click', disconnect);
els.btnStart.addEventListener('click', startExam);
els.btnStop.addEventListener('click', stopExam);

// ── INIT ──────────────────────────────────────────────────────────────────
log('ProctorAI frontend loaded. Click Connect to begin.', 'success');
setWsStatus('disconnected');
