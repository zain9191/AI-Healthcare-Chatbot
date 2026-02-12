// Global Variables
const chatBox = document.getElementById('chat-messages');
const inputField = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const suggestBox = document.getElementById('suggestions');
const voiceBtn = document.getElementById('voice-btn');

let step = 0;
let userName = "";
let mainSymptom = "";
let potentialDisease = "";
let finalSymptoms = [];
let voiceEnabled = false;
let isBotSpeaking = false;

// --- SPEECH RECOGNITION SETUP ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';

    // CHANGE 1: Enable Interim Results so you see text WHILE you talk
    recognition.interimResults = true;

    recognition.onstart = function () {
        console.log("Microphone active.");
        if (voiceEnabled) {
            voiceBtn.textContent = "Listening...";
            voiceBtn.classList.add("active");
            voiceBtn.style.backgroundColor = "#e74c3c";
        }
    };

    recognition.onresult = function (event) {
        // CHANGE 2: Handle partial results
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript;
        }

        console.log("Heard:", transcript);

        // Show text immediately in the box
        if (!isBotSpeaking) {
            inputField.value = transcript;
        }
    };

    recognition.onend = function () {
        // When you stop talking, we verify if we should send it or restart
        if (voiceEnabled && !isBotSpeaking) {
            // If we have text in the box, send it automatically!
            if (inputField.value.trim().length > 0) {
                handleInput();
            } else {
                // If silence, just restart listening
                try { recognition.start(); } catch (e) { }
            }
        } else {
            if (!voiceEnabled) {
                voiceBtn.textContent = "Voice: OFF";
                voiceBtn.style.backgroundColor = "transparent";
                voiceBtn.classList.remove("active");
            }
        }
    };

    recognition.onerror = function (event) {
        console.warn("Speech Error:", event.error);
        if (event.error === 'not-allowed') {
            alert("Microphone access blocked. Please click the lock icon in your address bar and Allow Microphone.");
            voiceEnabled = false;
            voiceBtn.textContent = "Voice: OFF";
        }
    };
} else {
    alert("Your browser does not support Voice. Please use Chrome.");
}

// --- VOICE TOGGLE ---
function toggleVoice() {
    if (!recognition) return;

    voiceEnabled = !voiceEnabled;

    if (voiceEnabled) {
        try { recognition.start(); } catch (e) { console.error(e); }
        // Don't use speak() here to avoid triggering the "bot speaking" lock immediately
    } else {
        voiceBtn.textContent = "Voice: OFF";
        voiceBtn.classList.remove("active");
        voiceBtn.style.backgroundColor = "transparent";
        recognition.stop();
        window.speechSynthesis.cancel();
    }
}

// --- TEXT TO SPEECH ---
function speak(text) {
    if (!voiceEnabled) return;

    isBotSpeaking = true;
    if (recognition) recognition.stop();

    const cleanText = text.replace(/<[^>]*>/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);

    utterance.onend = function () {
        isBotSpeaking = false;
        if (voiceEnabled && recognition) {
            try { recognition.start(); } catch (e) { }
        }
    };

    window.speechSynthesis.speak(utterance);
}

// --- CHAT LOGIC ---
function addMessage(text, sender, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    if (isHtml) div.innerHTML = text; else div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;

    if (sender === 'bot') speak(text);
}

async function handleInput() {
    const val = inputField.value.trim();
    if (!val) return;

    // Stop listening momentarily to process
    if (recognition) recognition.stop();

    addMessage(val, 'user');
    inputField.value = '';
    suggestBox.style.display = 'none';

    // Helper for Flask
    const postData = async (url, data) => {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return await response.json();
    };

    if (step === 4) {
        step = 0; userName = "";
        addMessage("Let's start over. What is your name?", 'bot');
        return;
    }

    if (step === 0) {
        userName = val;
        addMessage(`Hi ${userName}. Please describe your symptoms.`, 'bot');
        step = 1;
    } else if (step === 1) {
        const data = await postData('/check_pattern', { pattern: val });

        if (data.matches.length > 0) {
            mainSymptom = data.matches[0];
            finalSymptoms = [...data.matches];

            if (finalSymptoms.length > 1) {
                addMessage(`I noticed: ${finalSymptoms.join(", ")}.`, 'bot');
            } else {
                addMessage(`I understand you have ${mainSymptom}.`, 'bot');
            }

            const predData = await postData('/predict_initial', { symptom: mainSymptom });
            potentialDisease = predData.predicted_disease;

            const related = predData.related_symptoms;
            const newRelated = related.filter(s => !finalSymptoms.includes(s));

            if (newRelated.length > 0) {
                let html = `Do you have any of these?<br>`;
                newRelated.forEach(s => {
                    html += `<label class="option-label"><input type="checkbox" value="${s}"> ${s}</label>`;
                });
                html += `<button class="confirm-btn" onclick="confirmSymptoms()">Confirm Selected</button>`;
                addMessage(html, 'bot', true);
                step = 2;
            } else {
                addMessage("How many days have you been sick?", 'bot');
                step = 3;
            }
        } else {
            addMessage("I didn't catch that. Try 'fever' or 'cough'.", 'bot');
        }
    } else if (step === 3) {
        const days = parseInt(val);
        if (isNaN(days)) {
            addMessage("Please say just the number (e.g. 3).", 'bot');
            return;
        }

        addMessage("Analyzing...", 'bot');
        const finalData = await postData('/predict_final', {
            initial_disease: potentialDisease,
            symptoms_exp: finalSymptoms,
            days: days
        });

        let resultHtml = `<strong>Diagnosis:</strong> ${finalData.text}<br><br>`;
        resultHtml += `<strong>Details:</strong> ${finalData.description_present}<br><br>`;
        resultHtml += `<strong>Specialist:</strong> ${finalData.specialist_present}<br>`;
        resultHtml += `<em>${finalData.severity_message}</em>`;

        addMessage(resultHtml, 'bot', true);
        addMessage("Say anything to restart.", 'bot');
        step = 4;
    }
}

window.confirmSymptoms = function () {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
    checkboxes.forEach(cb => finalSymptoms.push(cb.value));
    addMessage(`Added ${checkboxes.length} symptoms.`, 'user');
    addMessage("How many days have you been sick?", 'bot');
    step = 3;
};

// Event Listeners
sendBtn.addEventListener('click', handleInput);
inputField.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleInput(); });