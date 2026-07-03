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
    elif style == "exciting":
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
    """
    Sequential state-machine parser.
    
    KEY FIX: Handles BOTH inline and newline voice tag switching.
    Old approach: split by newline first → failed for inline tags like:
      '[narrator] text1 [male] dialogue [narrator] text2' (all one line)
    
    New approach: tokenize the ENTIRE text into [TAG] and TEXT tokens,
    then walk through sequentially. Voice changes flush the buffer.
    Emotions reset on every voice change (not just newlines).
    """
    VOICE_TAGS   = {"narrator", "child", "kid", "male", "female", "kid_boy", "kid_girl"}
    EMOTION_TAGS = {"happy", "sad", "excited", "whisper", "laughter"}
    ALL_KNOWN    = VOICE_TAGS | EMOTION_TAGS
    
    # Tokenize: split on [TAG] keeping delimiters, then split text on sentence boundaries
    raw_tokens = re.split(r'(\[[a-zA-Z0-9_]+\])', text)
    
    parsed_segments = []
    current_voice   = "default"
    current_emotion = None   # ONE emotion at a time, reset on voice change
    pending_text    = ""     # Accumulated text for current voice+emotion
    
    def flush(txt, voice, emotion):
        """Save pending text as a segment if non-empty."""
        txt = txt.strip()
        if not txt:
            return
        # Split on sentence-ending punctuation + ellipsis for natural segments
        parts = re.split(r'(\.\.\.|[.!?।]+)', txt)
        temp = ""
        for p in parts:
            if not p:
                continue
            if re.match(r'^(\.\.\.|[.!?।]+)$', p):
                if temp:
                    parsed_segments.append({"text": temp.strip(), "voice": voice, "emotion": emotion})
                    temp = ""
                continue
            ps = p.strip()
            if not ps:
                continue
            if temp:
                parsed_segments.append({"text": temp.strip(), "voice": voice, "emotion": emotion})
            temp = ps
        if temp:
            parsed_segments.append({"text": temp.strip(), "voice": voice, "emotion": emotion})
    
    for token in raw_tokens:
        if not token:
            continue
        
        # Check if this is a [TAG]
        if token.startswith("[") and token.endswith("]"):
            tag_name = token[1:-1].lower()
            if tag_name not in ALL_KNOWN:
                # Unknown tag — treat as text
                pending_text += token
                continue
            
            if tag_name in VOICE_TAGS:
                # Voice switch — FLUSH current buffer first, then update state
                flush(pending_text, current_voice, current_emotion)
                pending_text = ""
                # Map kid_boy/kid_girl to canonical names
                if tag_name == "kid_boy":
                    current_voice = "kid_boy"
                elif tag_name == "kid_girl":
                    current_voice = "kid_girl"
                else:
                    current_voice = tag_name
                # Voice switch also resets the emotion
                current_emotion = None
            
            elif tag_name in EMOTION_TAGS:
                # Emotion tag — only applies to this voice block
                # Only update if not already set (first emotion wins per voice block)
                if current_emotion is None:
                    current_emotion = f"[{tag_name}]"
        else:
            # Regular text — accumulate
            # Newlines are sentence separators; convert to space for natural flow
            pending_text += token.replace("\n", " ")
    
    # Flush any remaining text
    flush(pending_text, current_voice, current_emotion)
    
    # Post-process: strip stray tags from text content, build final segments
    ALL_STRIP = {f"[{t}]" for t in ALL_KNOWN}
    final_segments = []
    
    for seg in parsed_segments:
        tc = seg["text"]
        for tag in ALL_STRIP:
            tc = tc.replace(tag, "")
        tc = " ".join(tc.split()).strip()
        if len(tc) < 3:  # Skip near-empty fragments
            continue
        
        prefix = seg["emotion"] or ""
        if prefix:
            tc = f"{prefix} {tc}"
        
        final_segments.append({"text": tc, "voice": seg["voice"]})
    
    return final_segments

@app.post("/api/chat")
async def chat_agent(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # Let's prompt the local Ollama LLM to act as the Audio Configurator Agent
    system_prompt = (
        "You are 'Audio Web Studio AI', a creative audio script writer and pipeline configurator.\n"
        "Write rich, emotionally expressive audio scripts and return synthesis parameters.\n"
        "Return ONLY a clean JSON object:\n"
        "{\n"
        "  \"reply\": \"Short friendly message about what you created.\",\n"
        "  \"text\": \"The full audio script. MANDATORY RULES:\n"
        "VOICE TAGS (switch speaker): [narrator] [male] [female] [child] [kid_boy] [kid_girl]\n"
        "EMOTION TAGS (add feeling): [happy] [sad] [excited] [whisper] [laughter]\n"
        "RULES:\n"
        "1. ALWAYS start with [narrator] for storytelling.\n"
        "2. Place ONE emotion tag IMMEDIATELY after a voice tag when that character speaks with feeling.\n"
        "   Example: [child] [happy] मुझे केले बहुत पसंद हैं!\n"
        "3. Characters must speak their OWN dialogue with a voice tag — never put dialogue under [narrator].\n"
        "4. For a 5-6 minute story, write at LEAST 20-30 lines with frequent voice switches.\n"
        "5. Vary emotions — use [happy], [excited], [sad], [whisper] at natural moments.\n"
        "6. Animals/characters: use [male] for large/elder animals, [female] for birds/gentle animals, [kid_boy] or [child] for young animals.\n"
        "7. HINDI STORIES: Write naturally in Hindi. Use [narrator] for story description, character voices for dialogue.\n"
        "EXAMPLE (Hindi multi-character animal story):\n"
        "[narrator] एक घने जंगल में तीन दोस्त रहते थे — शेर राजा, तोता मिठू, और हाथी भोला।\n"
        "[narrator] एक दिन सुबह शेर राजा ने गर्जना की —\n"
        "[male] [excited] आज हम सबको मिलकर जंगल की नदी पार करनी है!\n"
        "[narrator] तोता मिठू पेड़ की डाली पर बैठकर बोला —\n"
        "[female] [happy] वाह! मैं तो उड़कर पार कर लूंगी, लेकिन भोला का क्या?\n"
        "[narrator] हाथी भोला थोड़ा घबराया और धीरे से बोला —\n"
        "[male] [whisper] मुझे तैरना नहीं आता... क्या तुम लोग मेरी मदद करोगे?\n"
        "[narrator] शेर राजा ने प्यार से कहा —\n"
        "[male] [happy] बिल्कुल! दोस्त दोस्त के काम आते हैं!\",\n"
        "  \"language\": \"english\", \"hindi\", \"bengali\", \"tamil\", \"telugu\", \"marathi\", \"gujarati\", \"kannada\", \"malayalam\", or \"punjabi\",\n"
        "  \"gender\": \"male\", \"female\", \"kid_boy\", or \"kid_girl\",\n"
        "  \"style\": \"normal\", \"storytelling\", \"whisper\", \"exciting\", or \"poem\"\n"
        "}\n"
        "Do not include markdown, code blocks, or any text outside this JSON."
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
        
        # English Twinkle Twinkle Demo — NOTE: Twinkle is a RHYME, not a song.
        # It uses poem-style emotional recitation here (no [singing] tag — that lives in Music module)
        elif lang == "english" and wants_twinkle:
            parsed_config["text"] = (
                "[happy] Twinkle, twinkle, little star, How I wonder what you are!\n"
                "[happy] Up above the world so high, Like a diamond in the sky.\n"
                "[sad] When the blazing sun is gone, When he nothing shines upon,\n"
                "[happy] Then you show your little light, Twinkle, twinkle, all the night."
            )
            parsed_config["reply"] = "I've loaded 'Twinkle Twinkle Little Star' as a poem with happy and sad emotional tones. (For a sung version, use the Music Studio.)"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "poem"
        
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
    
    # Fixed guidance_scale for speech — singing uses its own settings in the Music module
    cfg_scale = 2.5
    
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
            # IMPORTANT: Make voices MAXIMALLY DISTINCT so characters sound different
            accent = "indian accent" if language != "english" else ""
            is_hindi = language == "hindi"
            
            if voice == "narrator":
                # Narrator: authoritative, opposite gender to default speaker for contrast
                if gender in ["kid_girl", "female"]:
                    seg_tokens = ["male", "middle-aged", "moderate pitch"]
                else:
                    seg_tokens = ["female", "middle-aged", "moderate pitch"]
            elif voice == "male":
                # Male character: deep adult male — low pitch gives clear contrast
                seg_tokens = ["male", "middle-aged", "low pitch"]
            elif voice == "female":
                # Female character: clear adult female
                seg_tokens = ["female", "young adult"]
            elif voice in ["child", "kid"]:
                # Generic child
                seg_tokens = ["male", "child", "high pitch"]
            elif voice == "kid_boy":
                # Boy child
                seg_tokens = ["male", "child", "high pitch"]
            elif voice == "kid_girl":
                # Girl child
                seg_tokens = ["female", "child", "very high pitch"]
            else:
                # Default: use base speaker
                seg_tokens = instruct.split(", ")


                
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
