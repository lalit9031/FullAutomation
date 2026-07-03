#!/usr/bin/env python3
"""
Music Studio Server — Dedicated singing/song pipeline
Runs on port 8006. Completely separate from Audio (speech) module on port 8005.

Pipeline:
  - OmniVoice singing model (ModelsLab/omnivoice-singing)
  - [singing] tag required for all segments
  - English only (Hindi avoided — produces Chinese-style melodies)
  - Per-line emotion reset: each lyric line gets ONE emotion ([happy] OR [sad])
  - Higher guidance_scale (3.5) to enforce melody rhythm
  - Shorter inter-segment pause (0.05s) for musical continuity
"""

import os, sys, time, re, json
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── AMD ROCm environment ──────────────────────────────────────────────────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import torch
import soundfile as sf

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Music Studio")

# ── Load OmniVoice Singing Model ──────────────────────────────────────────────
model = None

@app.on_event("startup")
async def load_model():
    global model
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    print("Loading OmniVoice singing model onto AMD RX 7900 XTX (ROCm)...")
    model = OmniVoice.from_pretrained(
        "ModelsLab/omnivoice-singing",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    print("✅ Music Studio: OmniVoice singing model loaded!")

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

# ── Singing-specific tag parser ───────────────────────────────────────────────
KNOWN_EMOTIONS = {"happy", "sad", "excited", "whisper", "laughter", "romantic", "longing"}
VALID_EMOTIONS = {f"[{e}]" for e in KNOWN_EMOTIONS}

def parse_lyrics_to_segments(lyrics: str):
    """
    Parse lyric script into OmniVoice-ready segments.
    
    Rules:
    - Every segment MUST have [singing] prefix (Music Studio only)
    - Each line = ONE emotion max (fresh slate per line)
    - Ellipsis (...) and line endings are segment boundaries
    - [singing] tags in input are stripped and auto-added back
    """
    lines = lyrics.strip().split("\n")
    segments = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Fresh emotion per line
        line_emotion = None
        
        # Extract tags first
        parts = re.split(r'(\[[a-zA-Z0-9_]+\])', line)
        text_pieces = []
        
        for part in parts:
            if not part.strip():
                continue
            if part.startswith("[") and part.endswith("]"):
                tag_name = part[1:-1].lower()
                if tag_name == "singing":
                    pass  # Skip — we always add [singing] automatically
                elif tag_name in KNOWN_EMOTIONS and line_emotion is None:
                    line_emotion = f"[{tag_name}]"  # Only first emotion counts
            else:
                text_pieces.append(part)
        
        line_text = " ".join(text_pieces).strip()
        if not line_text:
            continue
        
        # Split on ellipsis or sentence-ending punctuation
        sub_phrases = re.split(r'(\.{3,}|[.!?]+)', line_text)
        temp = ""
        
        for phrase in sub_phrases:
            if not phrase:
                continue
            if re.match(r'^(\.{3,}|[.!?]+)$', phrase):
                if temp:
                    segments.append({
                        "text": temp.strip(),
                        "emotion": line_emotion
                    })
                    temp = ""
                continue
            phrase_s = phrase.strip()
            if not phrase_s:
                continue
            if temp:
                segments.append({
                    "text": temp.strip(),
                    "emotion": line_emotion
                })
            temp = phrase_s
        
        if temp:
            segments.append({
                "text": temp.strip(),
                "emotion": line_emotion
            })
    
    # Build final text with [singing] + optional emotion prefix
    final = []
    for seg in segments:
        text = seg["text"]
        # Strip any leftover tags
        for tag in VALID_EMOTIONS | {"[singing]"}:
            text = text.replace(tag, "")
        text = " ".join(text.split()).strip()
        if not text:
            continue
        
        tags = ["[singing]"]
        if seg["emotion"]:
            tags.append(seg["emotion"])
        
        final.append(" ".join(tags) + " " + text)
    
    return final

# ── Ollama AI Chat Agent ───────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat_agent(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    system_prompt = (
        "You are 'Music Studio AI', a creative song-writing assistant that configures an English singing pipeline.\n"
        "Analyze the user's request and write a song lyric script in a structured tagged format.\n"
        "Return ONLY a clean JSON object:\n"
        "{\n"
        "  \"reply\": \"A short friendly message describing the song you've written.\",\n"
        "  \"lyrics\": \"The song lyrics script. Rules:\n"
        "1. Write each lyric line on a new line.\n"
        "2. Prefix each line with ONE emotion tag: [happy], [sad], [excited], [romantic], [longing].\n"
        "3. Do NOT use [singing] tag — it is added automatically.\n"
        "4. Write 4-8 lines for a complete song. Vary emotions to tell a story.\n"
        "5. EXAMPLES:\n"
        "   Love song:\n"
        "   [romantic] You are my sunshine, my only light...\n"
        "   [happy] Every morning I wake up to your smile...\n"
        "   [longing] When you are far, the stars remind me of you...\n"
        "   [romantic] Come back to me, fill my heart with your glow...\n"
        "   Sad song:\n"
        "   [sad] The rain falls softly on the empty street...\n"
        "   [longing] I remember your voice in every beat...\n"
        "   [sad] The cold wind whispers your name tonight...\n"
        "   [sad] How do I learn to live without your light...\",\n"
        "  \"mood\": \"happy\", \"sad\", \"romantic\", \"excited\", or \"longing\",\n"
        "  \"title\": \"Short song title\"\n"
        "}\n"
        "Do not include markdown, code blocks, or extra text outside this JSON."
    )
    
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:14b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "stream": False
        }, timeout=60)
        
        raw_res = r.json()["message"]["content"].strip()
        
        # Clean markdown if present
        if raw_res.startswith("```"):
            raw_res = re.sub(r"^```[a-z]*\n?", "", raw_res)
            raw_res = re.sub(r"\n?```$", "", raw_res)
        
        parsed = json.loads(raw_res)
        
        # Normalise — lyrics key required
        if "lyrics" not in parsed and "text" in parsed:
            parsed["lyrics"] = parsed.pop("text")
        
        return JSONResponse(content=parsed)
    except Exception as e:
        print(f"Ollama error: {e}")
        return JSONResponse(content={
            "reply": "Couldn't connect to Ollama. Here's a default love song to try!",
            "lyrics": (
                "[romantic] You are my sunshine, my only sunshine\n"
                "[happy] You make me happy when skies are gray\n"
                "[longing] You'll never know dear how much I love you\n"
                "[sad] Please don't take my sunshine away"
            ),
            "mood": "romantic",
            "title": "My Sunshine"
        })

# ── Generate Singing Audio ────────────────────────────────────────────────────
@app.post("/api/generate")
async def generate_music(request: Request):
    from omnivoice import OmniVoiceGenerationConfig
    
    data = await request.json()
    lyrics = data.get("lyrics", data.get("text", ""))
    gender = data.get("gender", "female")
    
    if not lyrics.strip():
        return JSONResponse(status_code=400, content={"error": "No lyrics provided."})
    
    # Parse into singing segments
    segments = parse_lyrics_to_segments(lyrics)
    
    if not segments:
        return JSONResponse(status_code=400, content={"error": "Could not parse lyrics into segments."})
    
    # Build instruct prompt (singing voice profile using valid OmniVoice tokens)
    if gender == "male":
        instruct = "male, middle-aged, moderate pitch"
    elif gender == "female":
        instruct = "female, young adult, high pitch"
    elif gender == "kid_boy":
        instruct = "male, child, high pitch"
    elif gender == "kid_girl":
        instruct = "female, child, very high pitch"
    else:
        instruct = "female, young adult"
    
    # Higher guidance for singing to enforce melody/rhythm
    config = OmniVoiceGenerationConfig(
        num_step=70,
        guidance_scale=3.5,
        denoise=True,
        postprocess_output=True
    )
    
    print(f"[Music Studio] Generating {len(segments)} lyric segments | Instruct: '{instruct}'")
    
    t0 = time.time()
    try:
        import numpy as np
        audio_segments = []
        
        for i, seg_text in enumerate(segments, 1):
            # Format for OmniVoice: space out punctuation to prevent cutoff
            fmt = seg_text.replace(",", " , ").replace(".", " . ")
            fmt = fmt.replace("!", " ! ").replace("?", " ? ")
            fmt = " ".join(fmt.split())
            if not fmt.endswith("."):
                fmt = fmt + " ."
            
            print(f"  Segment {i}/{len(segments)}: '{fmt}'")
            audio = model.generate(text=fmt, instruct=instruct, language="English", config=config)
            audio_data = audio[0].cpu().numpy() if hasattr(audio[0], "cpu") else audio[0]
            audio_segments.append(audio_data)
        
        sample_rate = 24000
        # Musical pause: shorter than speech (0.05s) for musical continuity
        pause = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
        
        merged = []
        for i, seg in enumerate(audio_segments):
            if i > 0:
                merged.append(pause)
            merged.append(seg)
        
        final_audio = np.concatenate(merged)
        
        filename = f"music_{int(time.time())}.wav"
        filepath = OUTPUT_DIR / filename
        sf.write(str(filepath), final_audio, sample_rate)
        
        elapsed = time.time() - t0
        print(f"[Music Studio] Generated {filename} in {elapsed:.1f}s")
        
        return JSONResponse(content={
            "filename": filename,
            "url": f"/outputs/{filename}",
            "elapsed": f"{elapsed:.1f}s",
            "segments": len(segments)
        })
    except Exception as e:
        print(f"[Music Studio] Generation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
