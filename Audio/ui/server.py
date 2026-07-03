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
        '  "text": "The script/story/rhyme to synthesize. IMPORTANT: If generating text in Hindi or other Indian languages, write authentic, natural, grammatically correct, and standard language. If the user asks for a child rhyme in Hindi, generate a classic or highly natural Hindi child rhyme (e.g., about chanda mama, rain, or butterflies) with proper flow. Never output broken grammar, literal English translations, or gibberish.",\n'
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
        
        # Hindi Singing
        if lang == "hindi" and any(k in msg_l for k in ["singing", "sing", "गाना", "गाओ", "संगीत"]):
            parsed_config["text"] = (
                "[singing] चंदा मामा दूर के, पुए पकाएं बूर के।\n"
                "[singing] आप खाएं थाली में, मुन्ने को दें प्याली में।\n"
                "[singing] प्याली गई टूट, मुन्ना गया रूठ।\n"
                "[singing] लाएंगे नई प्यालियां, बजा बजा के तालियां!\n"
                "[singing] उड़नखटोले बैठेंगे, मुन्ने राजा ऐठेंगे!"
            )
            parsed_config["reply"] = "मैंने बच्चों के लिए पूर्ण हिंदी बालगीत 'चंदा मामा दूर के' गायन [singing] शैली में लोड कर दिया है!"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "singing"
            
        # Hindi Poem / Rhymes (speech recitation, no singing tag)
        elif lang == "hindi" and any(k in msg_l for k in ["rhyme", "poem", "कविता", "बालगीत"]):
            parsed_config["text"] = (
                "चंदा मामा दूर के, पुए पकाएं बूर के।\n"
                "आप खाएं थाली में, मुन्ने को दें प्याली में।\n"
                "प्याली गई टूट, मुन्ना गया रूठ।\n"
                "लाएंगे नई प्यालियां, बजा बजा के तालियां!\n"
                "उड़नखटोले बैठेंगे, मुन्ने राजा ऐठेंगे!"
            )
            parsed_config["reply"] = "मैंने बच्चों के लिए सुंदर हिंदी कविता 'चंदा मामा दूर के' (सस्वर पाठ शैली) लोड कर दी है।"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "poem"
            
        # Hindi Stories
        elif lang == "hindi" and any(k in msg_l for k in ["story", "कहानी", "कथा"]):
            parsed_config["text"] = (
                "एक जंगल में एक छोटा सा बंदर रहता था, जिसका नाम था चीकू।\n"
                "चीकू बहुत नटखट था और उसे मीठे पके केले खाना बहुत पसंद था।\n"
                "एक दिन उसने एक बड़े पेड़ पर पीले-पीले केले लटके देखे और खुशी से उछल पड़ा!"
            )
            parsed_config["reply"] = "मैंने चीकू बंदर की एक मजेदार हिंदी कहानी लोड कर दी है।"
            parsed_config["style"] = "storytelling"
            
        # Hindi Whisper
        elif lang == "hindi" and style == "whisper":
            parsed_config["text"] = (
                "धीरे से बोलो। हवा में कुछ फुसफुसाहट है।\n"
                "क्या तुमने भी वह आवाज़ सुनी? कोई चुपके से आ रहा है।"
            )
            parsed_config["reply"] = "रहस्यमयी फुसफुसाहट (whisper) के लिए स्क्रिप्ट लोड कर दी गई है।"
            
        # English Singing
        elif lang == "english" and any(k in msg_l for k in ["singing", "sing", "song"]):
            parsed_config["text"] = (
                "[singing] Twinkle, twinkle, little star, How I wonder what you are!\n"
                "[singing] Up above the world so high, Like a diamond in the sky.\n"
                "[singing] When the blazing sun is gone, When he nothing shines upon,\n"
                "[singing] Then you show your little light, Twinkle, twinkle, all the night."
            )
            parsed_config["reply"] = "I've loaded the full English nursery rhyme 'Twinkle Twinkle Little Star' in singing [singing] mode!"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "singing"
            
        # English Poem / Rhymes (speech recitation, no singing tag)
        elif lang == "english" and any(k in msg_l for k in ["rhyme", "poem", "poetry"]):
            parsed_config["text"] = (
                "Twinkle, twinkle, little star, How I wonder what you are!\n"
                "Up above the world so high, Like a diamond in the sky.\n"
                "When the blazing sun is gone, When he nothing shines upon,\n"
                "Then you show your little light, Twinkle, twinkle, all the night."
            )
            parsed_config["reply"] = "I've loaded the English poem 'Twinkle Twinkle Little Star' in rhythmic poem mode!"
            parsed_config["gender"] = "kid_girl"
            parsed_config["style"] = "poem"
            
        # English Stories
        elif lang == "english" and any(k in msg_l for k in ["story", "fairy tale"]):
            parsed_config["text"] = (
                "Once upon a time, in a magical forest, lived a little golden squirrel.\n"
                "She loved gathering glowing acorns under the silver light of the moon."
            )
            parsed_config["reply"] = "I've set up a whimsical children's story in English."
            parsed_config["style"] = "storytelling"
            
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
    
    # 1. Parse text into phrases using newlines, major punctuation, and commas (with word-count merging)
    import re
    raw_lines = re.split(r'[\n.!?;]+', text)
    text_lines = []
    
    # Check for active emotion tags in the user prompt script
    has_happy = "[happy]" in text
    has_sad = "[sad]" in text
    has_excited = "[excited]" in text
    has_whisper = "[whisper]" in text
    
    for line in raw_lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Split line by commas
        comma_parts = [p.strip() for p in line_strip.split(",") if p.strip()]
        
        # Merge parts if they are too short (less than 3 words) to prevent choppy synthesis
        temp_phrase = ""
        for part in comma_parts:
            # Clean all bracketed tags from the text content to prevent duplicate synthesis
            part_clean = part
            for tag in ["[singing]", "[happy]", "[sad]", "[excited]", "[whisper]"]:
                part_clean = part_clean.replace(tag, "")
            part_clean = part_clean.strip()
            
            if not part_clean:
                continue
                
            if not temp_phrase:
                temp_phrase = part_clean
            else:
                word_count = len(temp_phrase.split())
                if word_count < 3:
                    temp_phrase = f"{temp_phrase}, {part_clean}"
                else:
                    # Construct correct prefix tags
                    tags = []
                    if style == "singing" or "[singing]" in text:
                        tags.append("[singing]")
                    if has_happy:
                        tags.append("[happy]")
                    elif has_sad:
                        tags.append("[sad]")
                    elif has_excited:
                        tags.append("[excited]")
                    elif has_whisper or style == "whisper":
                        tags.append("[whisper]")
                        
                    prefix = " ".join(tags)
                    if prefix:
                        temp_phrase = f"{prefix} {temp_phrase}"
                    text_lines.append(temp_phrase)
                    temp_phrase = part_clean
        if temp_phrase:
            tags = []
            if style == "singing" or "[singing]" in text:
                tags.append("[singing]")
            if has_happy:
                tags.append("[happy]")
            elif has_sad:
                tags.append("[sad]")
            elif has_excited:
                tags.append("[excited]")
            elif has_whisper or style == "whisper":
                tags.append("[whisper]")
                
            prefix = " ".join(tags)
            if prefix:
                temp_phrase = f"{prefix} {temp_phrase}"
            text_lines.append(temp_phrase)
        
    instruct = get_instruct_prompt(gender, style, language)
    
    print(f"Generating: Phrases={text_lines}, Instruct='{instruct}'")
    
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
        for i, line in enumerate(text_lines, 1):
            # Apply community-recommended formatting to prevent final-word cutoff bugs
            formatted_line = line.replace(",", " , ")
            formatted_line = formatted_line.replace(".", " . ")
            formatted_line = formatted_line.replace("!", " ! ")
            formatted_line = formatted_line.replace("?", " ? ")
            formatted_line = formatted_line.replace("।", " । ")
            formatted_line = " ".join(formatted_line.split())
            if not formatted_line.endswith(".") and not formatted_line.endswith("।"):
                formatted_line = f"{formatted_line} ."
                
            print(f"Generating segment {i}/{len(text_lines)}: '{formatted_line}' (original: '{line}')")
            audio = model.generate(text=formatted_line, instruct=instruct, config=config)
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
