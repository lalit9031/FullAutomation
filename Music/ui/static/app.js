// Music Studio — app.js

const EMOTION_ICONS = {
  happy: '😊', sad: '😢', excited: '🎉', romantic: '❤️',
  longing: '💫', whisper: '🤫', laughter: '😄'
};

let currentTitle = '';
let currentMood = '';

// ── Send chat message ──────────────────────────────────────────────
async function sendMessage(e) {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  appendMsg(msg, 'user');
  input.value = '';
  setInputDisabled(true);

  const typingId = appendTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();

    removeTyping(typingId);
    appendMsg(data.reply || 'Here are your lyrics!', 'ai');

    // Populate lyrics editor
    if (data.lyrics) {
      document.getElementById('lyrics-input').value = data.lyrics;
      updateSegmentPreview(data.lyrics);
    }

    // Update mood badge
    if (data.mood) {
      currentMood = data.mood;
      const icon = EMOTION_ICONS[data.mood] || '🎵';
      document.getElementById('mood-badge').textContent = `${icon} ${data.mood}`;
    }

    // Update title badge
    if (data.title) {
      currentTitle = data.title;
      document.getElementById('title-badge').textContent = data.title;
    }

  } catch (err) {
    removeTyping(typingId);
    appendMsg('Could not reach the AI agent. You can still write lyrics manually!', 'ai');
    showToast('⚠ Ollama not reachable', 'error');
  } finally {
    setInputDisabled(false);
    document.getElementById('chat-input').focus();
  }
}

// ── Generate Song ──────────────────────────────────────────────────
async function generateSong() {
  const lyrics = document.getElementById('lyrics-input').value.trim();
  if (!lyrics) {
    showToast('Please enter lyrics first!', 'error');
    return;
  }

  const gender = document.getElementById('select-gender').value;

  // Show progress
  document.getElementById('btn-generate').disabled = true;
  document.getElementById('progress-wrap').style.display = 'flex';
  document.getElementById('player').style.display = 'none';
  
  const bar = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  bar.style.width = '0%';
  label.textContent = 'Sending to OmniVoice...';

  // Fake progress animation
  let pct = 0;
  const fakeProgress = setInterval(() => {
    pct = Math.min(pct + Math.random() * 8, 88);
    bar.style.width = pct + '%';
  }, 600);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lyrics, gender })
    });

    clearInterval(fakeProgress);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Generation failed');
    }

    const data = await res.json();

    bar.style.width = '100%';
    label.textContent = `✅ Done in ${data.elapsed} — ${data.segments} segments`;

    setTimeout(() => {
      document.getElementById('progress-wrap').style.display = 'none';
      showPlayer(data.url, data.elapsed, data.segments);
    }, 600);

  } catch (err) {
    clearInterval(fakeProgress);
    document.getElementById('progress-wrap').style.display = 'none';
    showToast(`❌ ${err.message}`, 'error');
    console.error(err);
  } finally {
    document.getElementById('btn-generate').disabled = false;
  }
}

// ── Show audio player ──────────────────────────────────────────────
function showPlayer(url, elapsed, segments) {
  const playerEl = document.getElementById('player');
  const audioEl  = document.getElementById('audio-player');
  const titleEl  = document.getElementById('player-title');
  const timeEl   = document.getElementById('player-time');
  const dlBtn    = document.getElementById('btn-download');

  titleEl.textContent = currentTitle || 'Generated Song';
  timeEl.textContent  = `${elapsed} · ${segments} segments · OmniVoice Singing`;
  audioEl.src = url;
  dlBtn.href  = url;
  dlBtn.download = url.split('/').pop();

  playerEl.style.display = 'block';
  audioEl.play().catch(() => {});
  showToast('🎵 Song ready!', 'success');
}

// ── Segment preview ────────────────────────────────────────────────
document.getElementById('lyrics-input').addEventListener('input', (e) => {
  updateSegmentPreview(e.target.value);
});

function updateSegmentPreview(lyrics) {
  const lines = lyrics.split('\n').filter(l => l.trim());
  const container = document.getElementById('segments-list');
  const wrap = document.getElementById('segments-preview');

  if (!lines.length) { wrap.style.display = 'none'; return; }

  let chips = '';
  let segIdx = 1;

  lines.forEach(line => {
    const emotionMatch = line.match(/\[([a-z]+)\]/i);
    const emotion = emotionMatch ? emotionMatch[1].toLowerCase() : null;
    const text = line.replace(/\[[a-zA-Z0-9_]+\]/g, '').trim();
    
    // Split at ellipsis
    const subPhrases = text.split(/\.{3,}/).filter(p => p.trim());
    
    subPhrases.forEach(phrase => {
      phrase = phrase.trim();
      if (!phrase) return;
      const icon = emotion ? (EMOTION_ICONS[emotion] || '🎵') : '🎵';
      const emotionLabel = emotion ? `<span class="seg-emotion">${icon}[${emotion}]</span>` : '';
      chips += `<div class="segment-chip"><span class="seg-num">${segIdx++}</span>${emotionLabel}${phrase.substring(0, 35)}${phrase.length > 35 ? '…' : ''}</div>`;
    });
  });

  container.innerHTML = chips;
  wrap.style.display = lines.length > 0 ? 'block' : 'none';
}

// ── Chat helpers ───────────────────────────────────────────────────
function appendMsg(text, role) {
  const container = document.getElementById('chat-messages');
  const icon = role === 'ai' ? '🎵' : '🧑';
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${icon}</div>
    <div class="msg-bubble"><p>${escapeHtml(text)}</p></div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendTyping() {
  const container = document.getElementById('chat-messages');
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'msg msg-ai';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🎵</div>
    <div class="msg-bubble">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function setInputDisabled(v) {
  document.getElementById('chat-input').disabled = v;
  document.getElementById('btn-send').disabled = v;
}

// ── Toast ──────────────────────────────────────────────────────────
function showToast(message, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Init ──────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // Show segment preview for default placeholder content
  const lyrics = document.getElementById('lyrics-input');
  if (lyrics.value) updateSegmentPreview(lyrics.value);
});
