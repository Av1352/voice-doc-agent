// Constants mapped to DOM UI interactions 
const ws = new WebSocket('ws://localhost:8000/ws');
const statusDisplay = document.getElementById('status-display');
const queryText = document.getElementById('query-text');
const latStt = document.getElementById('lat-stt');
const latRet = document.getElementById('lat-ret');
const latFirst = document.getElementById('lat-first');
const latTot = document.getElementById('lat-tot');
const uploadInput = document.getElementById('pdf-upload');
const uploadStatus = document.getElementById('upload-status');
const micBtn = document.getElementById('mic-btn');
const micContainer = document.getElementById('mic-container');

// Session globals
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioContext;

// Expecting explicit binary returns over socket mapping chunk pipelines natively
ws.binaryType = 'arraybuffer'; 

ws.onopen = () => {
    console.log("Secure WebSocket connected seamlessly.");
};

ws.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
        // Event payload maps to audio synth streams received inbound
        setStatus('Speaking');
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        try {
            // Buffer decode to speaker routes mapped sequentially
            const audioBuffer = await audioContext.decodeAudioData(event.data);
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(0);
            
            source.onended = () => {
                setStatus('Idle');
            };
        } catch (e) {
            console.error("Audio decoding bounds failed:", e);
            setStatus('Idle');
        }
    } else {
        // Assume text/JSON message maps to standard query + timings pipeline payload
        try {
            const data = JSON.parse(event.data);
            if (data.query) {
                queryText.textContent = `"${data.query}"`;
            }
            if (data.timings) {
                latStt.textContent = `${data.timings.stt_ms?.toFixed(2) || '--'} ms`;
                latRet.textContent = `${data.timings.retrieval_ms?.toFixed(2) || '--'} ms`;
                latFirst.textContent = `${data.timings.first_sentence_ms?.toFixed(2) || '--'} ms`;
                latTot.textContent = `${data.timings.total_ms?.toFixed(2) || '--'} ms`;
            }
            if (data.error) {
                console.error("Endpoint encountered error loop:", data.error);
                queryText.textContent = "Error executing request matrix.";
                setStatus('Idle');
            }
        } catch (e) {
            console.error("JSON deserialization layout failed:", e);
        }
    }
};

ws.onclose = () => {
    setStatus('Offline');
};

function setStatus(statusText) {
    statusDisplay.textContent = statusText;
    
    // Style injections mapping UI bounds
    if (statusText === 'Speaking') {
        statusDisplay.style.color = 'var(--accent-hover)';
    } else if (statusText === 'Recording') {
        statusDisplay.style.color = '#ef4444'; // Bright Red
    } else if (statusText === 'Transcribing') {
        statusDisplay.style.color = 'var(--text-main)';
    } else {
        statusDisplay.style.color = 'var(--text-muted)';
    }
}

async function initContext() {
    if (!audioContext) {
         audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
}

// Map UserMedia microphone input routes across navigator interfaces
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            // Re-assembly to strict contiguous Blob structure 
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            
            if (ws.readyState === WebSocket.OPEN) {
                setStatus('Transcribing');
                ws.send(audioBlob);
            } else {
                console.warn("WebSocket stream unresponsive. Payload dropped.");
                setStatus('Offline');
            }
        };
    })
    .catch(err => {
        console.error("Microphone hardware hooks blocked:", err);
        statusDisplay.textContent = "Mic access blocked by browser!";
        statusDisplay.style.color = '#ef4444';
    });

function startRecording() {
    if (!isRecording && mediaRecorder && mediaRecorder.state === 'inactive') {
        initContext();
        isRecording = true;
        audioChunks = [];
        
        micContainer.classList.add('recording');
        setStatus('Recording');
        mediaRecorder.start();
    }
}

function stopRecording() {
    if (isRecording && mediaRecorder && mediaRecorder.state === 'recording') {
        isRecording = false;
        
        micContainer.classList.remove('recording');
        mediaRecorder.stop();
        // State cascades directly to `Transcribing` natively resolving the `onstop` hook bounds.
    }
}

// Mount exact hold-push sequences traversing Desktop Mouses + Mobile Touches
micBtn.addEventListener('mousedown', startRecording);
micBtn.addEventListener('mouseup', stopRecording);
micBtn.addEventListener('mouseleave', stopRecording); // End gracefully if cursor drifts away actively

micBtn.addEventListener('touchstart', (e) => {
    e.preventDefault(); 
    startRecording();
}, {passive: false});

micBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopRecording();
}, {passive: false});

// Manage explicit multipart backend REST endpoints for indexing routines sequentially
uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = "Parsing and encoding context bounds...";
    uploadStatus.style.color = 'var(--text-main)';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('http://localhost:8000/upload-document', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'indexed') {
            uploadStatus.textContent = `Attached: ${result.filename} - Encoded ${result.chunk_count} slices safely.`;
            uploadStatus.style.color = 'var(--accent-color)';
        } else {
            uploadStatus.textContent = `Error: ${result.message}`;
            uploadStatus.style.color = '#ef4444';
        }
    } catch (err) {
        uploadStatus.textContent = "Fatal: Endpoint disconnected during POST.";
        uploadStatus.style.color = '#ef4444';
        console.error(err);
    }
});
