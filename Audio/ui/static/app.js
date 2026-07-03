/* =============================================================================
   app.js — Audio Studio Client-side Logic & Agent Integration
   ============================================================================= */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const langPills = document.querySelectorAll("#lang-selector .pill");
    const genderCards = document.querySelectorAll("#gender-selector .grid-card");
    const styleCards = document.querySelectorAll("#style-selector .grid-card");
    const scriptText = document.getElementById("script-text");
    const btnGenerate = document.getElementById("btn-generate");

    // Chat Elements
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");

    // Output Elements
    const outputPlaceholder = document.getElementById("output-placeholder");
    const outputLoader = document.getElementById("output-loader");
    const playerBox = document.getElementById("player-box");
    const audioPlayer = document.getElementById("audio-player");
    const elapsedTime = document.getElementById("elapsed-time");
    const activeInstruct = document.getElementById("active-instruct");
    const btnDownload = document.getElementById("btn-download");

    // Current State
    let currentState = {
        language: "english",
        gender: "male",
        style: "normal",
        text: ""
    };

    // Initialize state from default active elements
    const syncStateFromUI = () => {
        const activeLang = document.querySelector("#lang-selector .pill.active");
        const activeGender = document.querySelector("#gender-selector .grid-card.active");
        const activeStyle = document.querySelector("#style-selector .grid-card.active");

        if (activeLang) currentState.language = activeLang.dataset.value;
        if (activeGender) currentState.gender = activeGender.dataset.value;
        if (activeStyle) currentState.style = activeStyle.dataset.value;
        currentState.text = scriptText.value;
    };
    syncStateFromUI();

    // ── Helper: Set UI state dynamically ────────────────────────────────────
    const setUIState = (lang, gender, style, text) => {
        // 1. Language
        langPills.forEach(pill => {
            if (pill.dataset.value === lang) {
                pill.classList.add("active");
            } else {
                pill.classList.remove("active");
            }
        });

        // 2. Gender
        genderCards.forEach(card => {
            if (card.dataset.value === gender) {
                card.classList.add("active");
            } else {
                card.classList.remove("active");
            }
        });

        // 3. Style
        styleCards.forEach(card => {
            if (card.dataset.value === style) {
                card.classList.add("active");
            } else {
                card.classList.remove("active");
            }
        });

        // 4. Text
        if (text) {
            scriptText.value = text;
        }

        // Sync local object
        currentState = { language: lang, gender, style, text: scriptText.value };
    };

    // ── Interactive UI Selectors ───────────────────────────────────────────
    const setupSelector = (elements, stateKey) => {
        elements.forEach(el => {
            el.addEventListener("click", () => {
                elements.forEach(sibling => sibling.classList.remove("active"));
                el.classList.add("active");
                currentState[stateKey] = el.dataset.value;
            });
        });
    };

    setupSelector(langPills, "language");
    setupSelector(genderCards, "gender");
    setupSelector(styleCards, "style");

    scriptText.addEventListener("input", (e) => {
        currentState.text = e.target.value;
    });

    // ── Chat Panel logic ───────────────────────────────────────────────────
    const addMessage = (sender, text) => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender === "user" ? "user-msg" : "agent-msg"}`;
        msgDiv.textContent = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // User message
        addMessage("user", text);
        chatInput.value = "";

        // Show typing indicator
        const typingDiv = document.createElement("div");
        typingDiv.className = "message agent-msg typing";
        typingDiv.textContent = "Writing pipeline parameters...";
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            const config = await res.json();

            // Remove typing
            typingDiv.remove();

            // Agent response
            addMessage("agent", config.reply);

            // Update parameters
            setUIState(config.language, config.gender, config.style, config.text);

        } catch (err) {
            console.error(err);
            typingDiv.remove();
            addMessage("agent", "Oops! I encountered an error communicating with the agent server.");
        }
    });

    // ── Generation Logic ────────────────────────────────────────────────────
    btnGenerate.addEventListener("click", async () => {
        // Sync text first
        currentState.text = scriptText.value.trim();
        if (!currentState.text) {
            alert("Please type some text to generate audio, or tell the AI agent to write it for you.");
            return;
        }

        // Show loading state
        outputPlaceholder.classList.add("hidden");
        playerBox.classList.add("hidden");
        outputLoader.classList.remove("hidden");
        btnGenerate.disabled = true;

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(currentState)
            });

            if (!res.ok) throw new Error("Audio synthesis failed");

            const data = await res.json();

            // Setup audio player
            audioPlayer.src = data.url;
            elapsedTime.textContent = `Elapsed: ${data.elapsed}`;
            activeInstruct.textContent = `Instruct: ${data.instruct}`;
            btnDownload.href = data.url;
            btnDownload.download = data.filename;

            // Show player
            outputLoader.classList.add("hidden");
            playerBox.classList.remove("hidden");

        } catch (err) {
            console.error(err);
            outputLoader.classList.add("hidden");
            outputPlaceholder.classList.remove("hidden");
            alert("Error generating audio. Check terminal logs for ROCm GPU warnings.");
        } finally {
            btnGenerate.disabled = false;
        }
    });
});
