import React, { useState, useEffect, useRef } from 'react';
import './index.css';

export default function App() {
  const [status, setStatus] = useState("Ready");
  const [statusType, setStatusType] = useState("idle");
  const [documentLoaded, setDocumentLoaded] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [transcript, setTranscript] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [latency, setLatency] = useState({
    stt: '--',
    vector: '--',
    ttfs: '--',
    total: '--'
  });

  const ws = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const audioContext = useRef(null);
  const nextPlayTime = useRef(0);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    
    ws.current.onopen = () => {
      console.log('WebSocket Connected');
      setStatus("Connected. Upload PDF first.");
      setStatusType("idle");
    };

    ws.current.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        if (msg.error) {
          setStatusType("error");
          setStatus("Error: " + msg.error);
        } else if (msg.query) {
          setTranscript(`"${msg.query}"`);
          if (msg.timings) {
            setLatency({
              stt: `${msg.timings.stt_ms.toFixed(0)}ms`,
              vector: `${msg.timings.retrieval_ms.toFixed(0)}ms`,
              ttfs: `${msg.timings.first_sentence_ms.toFixed(0)}ms`,
              total: `${msg.timings.total_ms.toFixed(0)}ms`
            });
          }
          setStatusType("success");
          setStatus("Ready for next question.");
        }
      } else {
        // Blob data (Audio bytes)
        const arrayBuffer = await event.data.arrayBuffer();
        if (!audioContext.current) {
          audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        try {
          const audioBuffer = await audioContext.current.decodeAudioData(arrayBuffer);
          playAudioBuffer(audioBuffer);
        } catch (e) {
          console.error("Audio decoding error:", e);
        }
      }
    };

    ws.current.onclose = () => {
      setStatus("Disconnected. Reconnecting...");
      setStatusType("error");
      setTimeout(connectWebSocket, 3000);
    };
  };

  const playAudioBuffer = (buffer) => {
    const source = audioContext.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.current.destination);
    
    const currentTime = audioContext.current.currentTime;
    if (nextPlayTime.current < currentTime) {
      nextPlayTime.current = currentTime;
    }
    
    source.start(nextPlayTime.current);
    nextPlayTime.current += buffer.duration;
  };

  const startRecording = async () => {
    if (status.includes('Disconnected') || status.includes('Reconnecting')) return;
    try {
      if (!audioContext.current) {
        audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.current.state === 'suspended') {
        await audioContext.current.resume();
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(audioBlob);
          setStatus("Processing audio...");
          setStatusType("processing");
          setLatency({ stt: '--', vector: '--', ttfs: '--', total: '--' });
        }
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setStatus("Listening...");
      setStatusType("recording");
      setTranscript("");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setStatus("Microphone access denied");
      setStatusType("error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setStatus(`Uploading ${file.name}...`);
    setStatusType("processing");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload-document", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok && data.status === "indexed") {
        setStatus(`Indexed ${data.chunk_count || 'document'} chunks successfully.`);
        setStatusType("success");
        setDocumentLoaded(true);
        setUploadedFilename(file.name);
      } else {
        setStatus(`Error: ${data.detail || data.message || 'Upload failed'}`);
        setStatusType("error");
      }
    } catch (err) {
      setStatus(`Failed to upload ${file.name}. Check backend connection.`);
      setStatusType("error");
    }
  };

  const getLatencyColor = (valueStr, threshold) => {
    if (valueStr === '--') return { color: 'var(--text-main)' };
    const val = parseFloat(valueStr);
    if (isNaN(val)) return { color: 'var(--text-main)' };
    return val < threshold ? { color: '#10b981' } : { color: '#ef4444' };
  };

  return (
    <div className="app-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h1 style={{ fontSize: '1.25rem', fontWeight: '700', letterSpacing: '-0.02em', margin: 0 }}>VoiceDoc AI</h1>
          <span className="label" style={{ fontSize: '0.7rem' }}>/ VOICE AGENT</span>
        </div>
        <div className="pill">Powered by Whisper + Claude + ElevenLabs</div>
      </header>

      <div className="content">
        <aside className="sidebar">
          <div>
            <div className="label" style={{ marginBottom: '16px' }}>// DOCUMENT</div>
            <label className="dropzone">
              <input 
                type="file" 
                accept="application/pdf" 
                style={{ display: 'none' }} 
                onChange={handleFileUpload} 
              />
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', color: 'var(--accent)' }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <p>Click to upload PDF</p>
            </label>
            {documentLoaded && (
              <div style={{ marginTop: '16px', textAlign: 'center' }}>
                <span className="pill" title={uploadedFilename} style={{ display: 'inline-block', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {uploadedFilename}
                </span>
              </div>
            )}
          </div>

          <div>
            <div className="label" style={{ marginBottom: '16px' }}>// HOW IT WORKS</div>
            <div className="steps">
              <div className="step">
                <span className="step-num">01</span>
                <span className="step-text">Upload a PDF document to be indexed in the vector database.</span>
              </div>
              <div className="step">
                <span className="step-num">02</span>
                <span className="step-text">Hold the microphone button and ask a question.</span>
              </div>
              <div className="step">
                <span className="step-num">03</span>
                <span className="step-text">AI synthesizes context and speaks the answer back.</span>
              </div>
              <div className="step">
                <span className="step-num">04</span>
                <span className="step-text">Monitor pipeline latencies in real-time.</span>
              </div>
            </div>
          </div>
        </aside>

        <main className="main-area">
          <div className="mic-container">
            <button 
              className={`mic-button ${isRecording ? 'recording' : ''}`}
              disabled={!documentLoaded}
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={(e) => { e.preventDefault(); if (!documentLoaded) return; startRecording(); }}
              onTouchEnd={(e) => { e.preventDefault(); if (!documentLoaded) return; stopRecording(); }}
            >
              <svg className="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
              </svg>
            </button>
            
            <div className="status-display">
              <div className={`status-text ${statusType}`}>{status}</div>
              <div className="transcript">{transcript}</div>
            </div>
          </div>

          <div className="latency-grid">
            <div className="latency-card">
              <div className="label">STT Recognition</div>
              <div className="latency-val" style={getLatencyColor(latency.stt, 500)}>{latency.stt}</div>
            </div>
            <div className="latency-card">
              <div className="label">Vector Retrieval</div>
              <div className="latency-val" style={getLatencyColor(latency.vector, 100)}>{latency.vector}</div>
            </div>
            <div className="latency-card">
              <div className="label">TTFS First Audio</div>
              <div className="latency-val" style={getLatencyColor(latency.ttfs, 1000)}>{latency.ttfs}</div>
            </div>
            <div className="latency-card">
              <div className="label">Total Pipeline</div>
              <div className="latency-val" style={getLatencyColor(latency.total, 2000)}>{latency.total}</div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
