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
    
    # ── Detect intent for content-type-specific instructions ──────────────────
    msg_lower = message.lower()
    is_poem   = any(k in msg_lower for k in ["poem", "rhyme", "poetry", "kavita", "कविता", "बालगीत"])
    is_story  = any(k in msg_lower for k in ["story", "kahani", "कहानी", "tale", "fairy"])
    is_long   = any(k in msg_lower for k in ["5 min", "6 min", "5-6 min", "long", "full length", "full story"])
    
    # Build content-type-specific guidance block
    if is_poem:
        content_rules = (
            "CONTENT TYPE: POEM / RHYME\n"
            "CRITICAL RULE: A poem uses ONE speaker voice only. NEVER use [narrator] in a poem.\n"
            "Use ONLY emotion tags to color each stanza/line. The single voice IS the poet.\n"
            "Format: ONE emotion tag per line → the voice carries the whole poem.\n"
            "Emotions flow like: [happy] → [excited] → [whisper] → [sad] → [happy]\n"
            "Each line = 1 poetic line. Put emotion tag at start of each stanza or when feeling changes.\n"
            "POEM EXAMPLE (English):\n"
            "[happy] The morning sun rises golden and bright,\n"
            "[happy] Painting the sky with colors of light.\n"
            "[excited] The birds sing and the rivers flow free,\n"
            "[whisper] In the quiet of dawn, there is only me.\n"
            "[sad] When evening comes and shadows grow long,\n"
            "[sad] I remember old dreams and a half-forgotten song.\n"
            "[happy] But the stars bring hope as they fill up the night,\n"
            "[happy] And morning will come again, pure and bright.\n"
            "HINDI POEM EXAMPLE:\n"
            "[happy] उगता सूरज लाल-नारंगी, चिड़ियाँ गाएं गीत।\n"
            "[happy] खेतों में हरियाली छाई, मन हो गया प्रीत।\n"
            "[excited] बच्चे दौड़े, खिलखिलाए, झूला झूलें संग।\n"
            "[whisper] शाम ढली तो माँ की लोरी, भर दे मन में रंग।\n"
            "[sad] पर जब बादल घिर आते हैं, आँखें भर आती हैं।\n"
            "[happy] फिर भी उम्मीद का दीपक, मन में जलता जाता है।\n"
            "SET: style=poem, gender=female (default for poems)\n"
        )
    elif is_story:
        length_note = (
            "LENGTH: User wants 5-6 minutes of audio. Write a FULL story with at LEAST 30-40 lines.\n"
            "Include: introduction, rising action, conflict/challenge, resolution, moral.\n"
            "Add multiple dialogue exchanges between characters — not just one line each.\n"
            if is_long else
            "Write a complete story with beginning, middle and end. Minimum 15-20 lines.\n"
        )
        content_rules = (
            "CONTENT TYPE: STORY / TALE\n"
            "Use [narrator] for all story description/narration.\n"
            "Use [male], [female], [child], [kid_boy], [kid_girl] for character dialogue.\n"
            "ALWAYS put ONE emotion tag after the character voice tag for dialogue.\n"
            "Animals: [male]=large/wise animals, [female]=gentle/bird animals, [child]/[kid_boy]=young animals.\n"
            + length_note +
            "STORY EXAMPLE (short excerpt):\n"
            "[narrator] एक घने जंगल में शेर राजा, तोता मिठू और हाथी भोला रहते थे।\n"
            "[narrator] एक दिन शेर राजा ने ऊँची आवाज़ में कहा —\n"
            "[male] [excited] आज हम नदी पार करेंगे, कोई नहीं रुकेगा!\n"
            "[narrator] तोता मिठू डर गई और बोली —\n"
            "[female] [whisper] लेकिन राजा जी, नदी में मगरमच्छ भी हैं!\n"
            "[narrator] हाथी भोला ने हिम्मत दिखाई और बोला —\n"
            "[male] [happy] मैं सबको अपनी पीठ पर बिठाकर पार करूंगा!\n"
            "[narrator] सभी जानवर खुश हो गए और हाथी पर सवार हो नदी पार की।\n"
            "SET: style=storytelling\n"
        )
    else:
        content_rules = (
            "CONTENT TYPE: NORMAL SPEECH / GENERAL\n"
            "Use simple emotion tags inline. No narrator needed unless it's a mini-story.\n"
            "Keep it natural and conversational.\n"
        )

    system_prompt = (
        "You are 'Audio Web Studio AI', an expert audio script writer for a TTS pipeline.\n"
        "Your job: detect what type of content the user wants, write a perfect tagged script, and return parameters.\n\n"
        + content_rules + "\n"
        "AVAILABLE TAGS:\n"
        "  Voice (story only): [narrator] [male] [female] [child] [kid_boy] [kid_girl]\n"
        "  Emotion: [happy] [sad] [excited] [whisper] [laughter]\n\n"
        "GOLDEN RULES:\n"
        "  - POEM: NO [narrator]. Single voice. Emotion tags only.\n"
        "  - STORY: [narrator] + character voices. Characters always get emotion tags.\n"
        "  - Never mix poem style with story style.\n"
        "  - One emotion tag per line maximum.\n\n"
        "Return ONLY this exact JSON (no markdown, no extra text):\n"
        "{\n"
        "  \"reply\": \"Short friendly message about what you created (1-2 sentences).\",\n"
        "  \"text\": \"<The full tagged script here>\",\n"
        "  \"language\": \"hindi\" or \"english\" or \"bengali\" etc,\n"
        "  \"gender\": \"male\" or \"female\" or \"kid_boy\" or \"kid_girl\",\n"
        "  \"style\": \"poem\" or \"storytelling\" or \"normal\" or \"whisper\" or \"exciting\"\n"
        "}"
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
