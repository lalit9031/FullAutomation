import os
import sys
import json
import time
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Force AMD ROCm env settings
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import soundfile as sf
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

app = FastAPI()

# Global reference to model
model = None

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

# Mount static directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve UI index.html
@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.on_event("startup")
def startup_event():
    global model
    print("Loading ModelsLab/omnivoice-singing Model onto AMD RX 7900 XTX (ROCm)...")
    try:
        model = OmniVoice.from_pretrained(
            "ModelsLab/omnivoice-singing", 
            device_map="cuda:0", 
            dtype=torch.float32
        )
        print("✅ ModelsLab/omnivoice-singing loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load OmniVoice Singing: {e}")

# Helper: Map simple frontend params to OmniVoice whitelisted instruct words
def get_instruct_prompt(gender: str, style: str, language: str) -> str:
    tokens = []
    
    # 1. Gender / Age
    if gender == "male":
        tokens.append("male")
    elif gender == "female":
        tokens.append("female")
    elif gender == "kid_boy":
        tokens.extend(["male", "child", "high pitch"])
    elif gender == "kid_girl":
        tokens.extend(["female", "child", "high pitch"])
        
    # 2. Style
    if style == "whisper":
        tokens.append("whisper")
    elif style == "exciting" or style == "singing":
        if "high pitch" not in tokens:
            tokens.append("high pitch")
    elif style == "calm":
        if "low pitch" not in tokens:
            tokens.append("low pitch")
    elif style == "storytelling" or style == "poem":
        if "child" not in tokens:
            tokens.append("middle-aged")  # Good narration tone for adults
        
    # 3. Indian accent mapping for non-English Indian languages
    if language != "english":
        tokens.append("indian accent")
        
    # Deduplicate keeping order
    seen = set()
    final = [x for x in tokens if not (x in seen or seen.add(x))]
    return ", ".join(final)

def parse_script_to_phrases(text: str, style: str, gender: str, language: str):
    import re
    is_singing = (style == "singing" or "[singing]" in text) and language == "english"
    
    # ── Pre-process: split on newlines first to track line boundaries ──
    # Each line gets its OWN emotion state — emotions do NOT carry across lines.
    # This prevents [happy] from line 1 bleeding into [excited] on line 2,
    # which causes OmniVoice to read tag names aloud.
    lines = text.split("\n")
    
    parsed_segments = []
    current_voice = "default"  # Voice CAN carry across lines (narrator stays narrator)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Each line starts with a FRESH emotion slate
        current_emotions = []
        
        # Split line into tags + text parts
        parts = re.split(r'(\[[a-zA-Z0-9_]+\])', line)
        
        # First pass: extract leading tags and text pieces
        text_pieces = []
        for part in parts:
            if not part.strip():
                continue
            if part.startswith("[") and part.endswith("]"):
                tag_name = part[1:-1].lower()
                if tag_name in ["narrator", "child", "kid", "male", "female"]:
                    current_voice = tag_name
                    # Voice switch also resets emotions
                    current_emotions = []
                elif tag_name in ["singing", "happy", "sad", "excited", "whisper", "laughter"]:
                    if part not in current_emotions:
                        current_emotions.append(part)
            else:
                text_pieces.append(part)
        
        # Combine remaining text, then split on sentence boundaries + ellipsis
        line_text = " ".join(text_pieces).strip()
        if not line_text:
            continue
        
        # Split on: newline, period, !, ?, ;, Hindi full stop, OR ellipsis (...)
        # Keep the delimiter so we can append it to the segment
        sentences = re.split(r'(\.{3,}|[.!?;।]+)', line_text)
        
        temp_text = ""
        for s in sentences:
            if not s:
                continue
            # Punctuation-only separator
            if re.match(r'^(\.{3,}|[.!?;।]+)$', s):
                if temp_text:
                    parsed_segments.append({
                        "text": temp_text.strip(),
                        "voice": current_voice,
                        "emotions": list(current_emotions)
                    })
                    temp_text = ""
                continue
            s_strip = s.strip()
            if not s_strip:
                continue
            if temp_text:
                parsed_segments.append({
                    "text": temp_text.strip(),
                    "voice": current_voice,
                    "emotions": list(current_emotions)
                })
            temp_text = s_strip
        
        if temp_text:
            parsed_segments.append({
                "text": temp_text.strip(),
                "voice": current_voice,
                "emotions": list(current_emotions)
            })
        
        # ── DO NOT carry emotions to next line ── (already handled by per-line reset above)
        # current_emotions is local to this loop iteration
        
        # Dummy reference to suppress unused-variable warning
            
    # Post-process parsed_segments:
    # 1. Strip any stray tag text from the segment content
    # 2. Prepend correct emotion prefix (max ONE primary emotion tag + [singing])
    final_segments = []
    EMOTION_TAGS = {"[singing]", "[happy]", "[sad]", "[excited]", "[whisper]", "[laughter]"}
    VOICE_TAGS   = {"[narrator]", "[child]", "[kid]", "[male]", "[female]"}
    ALL_STRIP    = EMOTION_TAGS | VOICE_TAGS
    
    for seg in parsed_segments:
        text_clean = seg["text"]
        for tag in ALL_STRIP:
            text_clean = text_clean.replace(tag, "")
        text_clean = " ".join(text_clean.split()).strip()
        
        if not text_clean:
            continue
        
        # Build tag prefix — use at most ONE non-[singing] emotion to avoid confusion
        seg_emotions = seg["emotions"]
        
        # Separate [singing] from other emotions
        non_singing = [e for e in seg_emotions if e != "[singing]"]
        has_singing = "[singing]" in seg_emotions or is_singing
        
        # Take only the FIRST non-singing emotion (e.g., [happy] OR [excited], never both)
        primary_emotion = non_singing[:1]  # At most one emotion tag
        
        tags = []
        if has_singing:
            tags.append("[singing]")
        tags.extend(primary_emotion)
        
        prefix = " ".join(tags)
        if prefix:
            text_clean = f"{prefix} {text_clean}"
            
        final_segments.append({
            "text": text_clean,
            "voice": seg["voice"]
        })
        
    return final_segments

@app.post("/api/chat")
async def chat_agent(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # Let's prompt the local Ollama LLM to act as the Audio Configurator Agent
    system_prompt = (
        "You are 'Audio Web Studio AI Assistant', a helpful local AI agent that configures an audio generation pipeline.\n"
        "Analyze the user's intent to set appropriate synthesis parameters.\n"
        "Return ONLY a clean JSON object containing:\n"
        '{\n'
        '  "reply": "A friendly message explaining how you updated the options based on their request.",\n'
        '  "text": "The script/story/rhyme to synthesize. IMPORTANT RULES FOR TAGS:\n'
        '1. Use emotion tags: [happy], [sad], [excited], [whisper], [laughter] dynamically to enhance the vocal delivery.\n'
        '2. Place these tags in-line at transition boundaries to match the narrative feeling (e.g., [happy] for joy, [excited] for surprises, [whisper] for secrets, [laughter] for funny parts). Do not mix too many tags, keep it natural.\n'
        '3. ENGLISH SINGING: For English rhymes/songs, combine [singing] with an emotion, e.g. \\"[singing] [happy] Twinkle, twinkle, little star...\\".\n'
        '4. HINDI/INDIAN LANGUAGES: Never use the [singing] tag for Hindi or other Indian languages because the singing model generates Chinese melodies. Instead, for Hindi singing/poem/rhymes, use rhythmic text with emotion tags (like [happy] or [excited]) without the [singing] tag.\n'
        '5. DUAL-VOICE STORYTELLING: For stories/dialogues, you can write character voice tags in-line to switch speakers. For kids stories, alternate between [narrator] (for the story teller) and [child] (for characters). For adult stories, alternate between [male] and [female].\n'
        '6. EXAMPLES:\n'
        '- English Kids Story: \\"[narrator] Once upon a time, a little girl said: [child] [happy] I found a shiny shell! [narrator] Then her father replied: [male] That is beautiful, sweetheart!\\"\n'
        '- Hindi Kids Story: \\"[narrator] एक जंगल में चीकू बंदर रहता था। एक दिन वह बोला: [child] [happy] आज तो मैं पके केले खाऊंगा! [narrator] तभी भालू दादा बोले: [male] मेरे पास भी एक केला है।\\"",\n'
        '  "language": "english", "hindi", "bengali", "tamil", "telugu", "marathi", "gujarati", "kannada", "malayalam", or "punjabi",\n'
        '  "gender": "male", "female", "kid_boy", or "kid_girl",\n'
        '  "style": "normal", "storytelling", "whisper", "exciting", "singing", or "poem"\n'
        '}\n'
        "Do not include markdown triple-ticks, code blocks, or extra text outside this JSON."
    )
    
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:14b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "stream": False
        })
        
        raw_res = r.json()["message"]["content"].strip()
        
        # Clean potential markdown output
        if raw_res.startswith("```"):
            raw_res = raw_res.strip("`").replace("json\n", "", 1).strip()
            
        parsed_config = json.loads(raw_res)
        
        # ── Python Intercept for Premium Native Scripts ────────────────────
        msg_l = message.lower()
        lang = parsed_config.get("language", "english")
        gender = parsed_config.get("gender", "male")
        style = parsed_config.get("style", "normal")
        
        # ── TEMPLATE LIBRARY (only fires when user explicitly asks for the exact template) ──
        # These are rich pre-crafted scripts. If the user wants custom content
        # (e.g., "love song", "horror story"), the LLM's dynamic JSON is used as-is.
        
        # Check if user is asking for specific known templates:
        wants_chanda_mama = any(k in msg_l for k in ["chanda mama", "chandamama", "चंदा मामा"])
        wants_twinkle = any(k in msg_l for k in ["twinkle twinkle", "twinkle star"])
        wants_chiku = any(k in msg_l for k in ["chiku", "चीकू", "chiku monkey", "monkey story"])
        wants_squirrel = "squirrel" in msg_l and "magic" in msg_l
        wants_whisper_demo = style == "whisper" and len(parsed_config.get("text", "")) < 10
        
        # Hindi Singing / Demo — Chanda Mama
        if lang == "hindi" and wants_chanda_mama and any(k in msg_l for k in ["singing", "sing", "गाना", "गाओ", "संगीत", "rhyme", "poem", "कविता", "बालगीत"]):
            parsed_config["text"] = (
                "[happy] चंदा मामा दूर के, पुए पकाएं बूर के।\n"
                "[happy] आप खाएं थाली में, मुन्ने को दें प्याली में।\n"
                "[happy] प्याली गई टूट, मुन्ना गया रूठ।\n"
                "[excited] लाएंगे नई प्यालियां, बजा बजा के तालियां!\n"
                "[happy] उड़नखटोले बैठेंगे, मुन्ने राजा ऐठेंगे!"
            )
            parsed_config["reply"] = "मैंने हिंदी बालगीत 'चंदा मामा दूर के' को [happy] और [excited] भावों के साथ लोड किया है। (नोट: हिंदी के लिए [singing] टैग का उपयोग नहीं होता क्योंकि यह चीनी राग उत्पन्न करता है।)"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "poem"
        
        # Hindi Chiku Monkey Story Demo
        elif lang == "hindi" and wants_chiku:
            parsed_config["text"] = (
                "[happy] एक जंगल में एक छोटा सा बंदर रहता था, जिसका नाम था चीकू।\n"
                "[happy] चीकू बहुत नटखट था और उसे मीठे पके केले खाना बहुत पसंद था।\n"
                "[excited] एक दिन उसने एक बड़े पेड़ पर पीले-पीले केले लटके देखे और खुशी से उछल पड़ा!"
            )
            parsed_config["reply"] = "मैंने चीकू बंदर की मजेदार हिंदी कहानी को [happy] और [excited] भावों के साथ लोड किया है।"
            parsed_config["style"] = "storytelling"
        
        # Hindi Whisper Demo (only if text is empty / not set by LLM)
        elif lang == "hindi" and wants_whisper_demo:
            parsed_config["text"] = (
                "[whisper] धीरे से बोलो। हवा में कुछ फुसफुसाहट है।\n"
                "[whisper] क्या तुमने भी वह आवाज़ सुनी? कोई चुपके से आ रहा है।"
            )
            parsed_config["reply"] = "रहस्यमयी फुसफुसाहट (whisper) के लिए स्क्रिप्ट लोड कर दी गई है।"
        
        # English Twinkle Twinkle Demo (only when explicitly requested)
        elif lang == "english" and wants_twinkle:
            parsed_config["text"] = (
                "[singing] [happy] Twinkle, twinkle, little star, How I wonder what you are!\n"
                "[singing] [happy] Up above the world so high, Like a diamond in the sky.\n"
                "[singing] [sad] When the blazing sun is gone, When he nothing shines upon,\n"
                "[singing] [happy] Then you show your little light, Twinkle, twinkle, all the night."
            )
            parsed_config["reply"] = "I've loaded the full 'Twinkle Twinkle Little Star' nursery rhyme in [singing] mode!"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "singing"
        
        # English Magic Squirrel Story Demo
        elif lang == "english" and wants_squirrel:
            parsed_config["text"] = (
                "[happy] Once upon a time, in a magical forest, lived a little golden squirrel.\n"
                "[excited] Suddenly, she spotted a glowing acorn under the silver light of the moon!\n"
                "[whisper] She walked closer quietly to investigate the magical spark..."
            )
            parsed_config["reply"] = "I've loaded the magic golden squirrel story with happy, excited, and whisper tones!"
            parsed_config["style"] = "storytelling"
        
        # ── ALL OTHER REQUESTS: Trust LLM's generated text/style/gender directly ──
        # This covers love songs, romantic poems, horror stories, custom content, etc.
        # The LLM JSON output from Qwen is used without override.
            
        return JSONResponse(content=parsed_config)
    except Exception as e:
        print(f"Ollama parsing failed: {e}")
        # Return fallback configuration
        return JSONResponse(content={
            "reply": "I heard you, but I couldn't connect to Ollama. I've set standard defaults for your generation.",
            "text": "चंदा मामा दूर के, पुए पकाएं बूर के।",
            "language": "hindi",
            "gender": "kid_girl",
            "style": "exciting"
        })

@app.post("/api/generate")
async def generate_audio(request: Request):
    global model
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model is not loaded."})
        
    data = await request.json()
    text = data.get("text", "")
    language = data.get("language", "english")
    gender = data.get("gender", "male")
    style = data.get("style", "normal")
    
    # 1. Parse text using our smart tag-aware sentence tokenizer
    text_lines = parse_script_to_phrases(text, style, gender, language)
        
    instruct = get_instruct_prompt(gender, style, language)
    
    print(f"Generating: Phrases={text_lines}, BaseInstruct='{instruct}'")
    
    # Increase guidance_scale to 3.5 for singing to enforce rhythm tags
    cfg_scale = 3.5 if (style == "singing" or "[singing]" in text) else 2.5
    
    config = OmniVoiceGenerationConfig(
        num_step=70,  # Optimal quality steps
        guidance_scale=cfg_scale,
        denoise=True,
        postprocess_output=True
    )
    
    t0 = time.time()
    try:
        import numpy as np
        
        if not text_lines:
            return JSONResponse(status_code=400, content={"error": "Text script is empty."})
            
        audio_segments = []
        for i, item in enumerate(text_lines, 1):
            line = item["text"]
            voice = item["voice"]
            
            # Resolve segment-specific instruct prompt based on voice type
            accent = "indian accent" if language != "english" else ""
            
            # Calculate voice instruct tokens
            if voice == "male":
                seg_tokens = ["male"]
                if "child" not in instruct:
                    seg_tokens.append("middle-aged")
            elif voice == "female":
                seg_tokens = ["female"]
                if "child" not in instruct:
                    seg_tokens.append("middle-aged")
            elif voice == "child":
                # Default to female child for kid_girl, male child for kid_boy
                g = "female" if (gender == "kid_girl" or "female" in instruct) else "male"
                seg_tokens = [g, "child", "high pitch"]
            elif voice == "narrator":
                # Narrator voice (usually adult, opposite to child/default profile)
                g = "male" if "female" in instruct else "female"
                seg_tokens = [g, "middle-aged"]
            else:
                # Fallback to default speaker configuration
                seg_tokens = [instruct]
                
            # Apply style properties
            if "[whisper]" in line:
                seg_tokens.append("whisper")
            if accent and accent not in seg_tokens:
                seg_tokens.append(accent)
                
            # Build clean deduplicated instruct string
            seen = set()
            final_tokens = []
            for t_group in seg_tokens:
                for t in t_group.split(","):
                    t = t.strip()
                    if t and t not in seen:
                        seen.add(t)
                        final_tokens.append(t)
            seg_instruct = ", ".join(final_tokens)
            
            # Apply community-recommended formatting to prevent final-word cutoff bugs
            formatted_line = line.replace(",", " , ")
            formatted_line = formatted_line.replace(".", " . ")
            formatted_line = formatted_line.replace("!", " ! ")
            formatted_line = formatted_line.replace("?", " ? ")
            formatted_line = formatted_line.replace("।", " । ")
            formatted_line = " ".join(formatted_line.split())
            if not formatted_line.endswith(".") and not formatted_line.endswith("।"):
                formatted_line = f"{formatted_line} ."
                
            print(f"Generating segment {i}/{len(text_lines)}: '{formatted_line}' | Voice='{voice}' | Instruct='{seg_instruct}'")
            audio = model.generate(text=formatted_line, instruct=seg_instruct, language=language.title(), config=config)
            audio_data = audio[0].cpu().numpy() if hasattr(audio[0], "cpu") else audio[0]
            audio_segments.append(audio_data)
            
        # Concatenate audio segments with a 0.2s natural pause between sentences
        sample_rate = 24000
        pause_samples = int(0.2 * sample_rate)
        pause_interval = np.zeros(pause_samples, dtype=np.float32)
        
        final_audio = []
        for i, segment in enumerate(audio_segments):
            if i > 0:
                final_audio.append(pause_interval)
            final_audio.append(segment)
            
        merged_audio = np.concatenate(final_audio)
        
        filename = f"gen_{int(time.time())}.wav"
        filepath = OUTPUT_DIR / filename
        sf.write(str(filepath), merged_audio, sample_rate)
        
        elapsed = time.time() - t0
        print(f"Generated complete audio {filename} in {elapsed:.1f}s")
        
        return JSONResponse(content={
            "filename": filename,
            "url": f"/outputs/{filename}",
            "elapsed": f"{elapsed:.1f}s",
            "instruct": instruct
        })
    except Exception as e:
        print(f"Generation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
