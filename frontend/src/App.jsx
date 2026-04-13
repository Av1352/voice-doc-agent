import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

function apiOrigin() {
  const base = import.meta.env.VITE_API_ORIGIN || window.location.origin;
  return String(base).replace(/\/$/, '');
}

function wsUrl() {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  const u = new URL(apiOrigin());
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  u.pathname = '/ws';
  u.search = '';
  u.hash = '';
  return u.toString();
}

export default function App() {
  const [status, setStatus] = useState("Connecting to server...");
  const [statusType, setStatusType] = useState("idle");
  const [documentLoaded, setDocumentLoaded] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [transcript, setTranscript] = useState("");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [aiResponse, setAiResponse] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [latency, setLatency] = useState({ stt: '--', vector: '--', ttfs: '--', total: '--' });

  const ws = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const audioCtx = useRef(null);
  const nextPlayTime = useRef(0);
  const activeSources = useRef(0);
  const recognition = useRef(null);
  const micBtn = useRef(null);
  const docLoadedRef = useRef(false);
  const retryCount = useRef(0);
  const retryTimer = useRef(null);
  /** @type {React.MutableRefObject<ArrayBuffer[]>} */
  const pendingAudioParts = useRef([]);

  useEffect(() => { docLoadedRef.current = documentLoaded; }, [documentLoaded]);

  // Touch events
  useEffect(() => {
    const btn = micBtn.current;
    if (!btn) return;
    const onTouchStart = (e) => {
      e.preventDefault();
      if (!docLoadedRef.current) return;
      handleMouseDown({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
    };
    const onTouchEnd = (e) => { e.preventDefault(); stopRecording(); };
    btn.addEventListener('touchstart', onTouchStart, { passive: false });
    btn.addEventListener('touchend', onTouchEnd, { passive: false });
    return () => {
      btn.removeEventListener('touchstart', onTouchStart);
      btn.removeEventListener('touchend', onTouchEnd);
    };
  }, [documentLoaded]);

  // WebSocket
  const connect = useCallback(() => {
    if (ws.current && (
      ws.current.readyState === WebSocket.OPEN ||
      ws.current.readyState === WebSocket.CONNECTING
    )) return;

    const socket = new WebSocket(wsUrl());
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
      retryCount.current = 0;
      pendingAudioParts.current = [];
      setStatus(docLoadedRef.current ? "Ready. Hold mic to speak." : "Upload a PDF to begin.");
      setStatusType("idle");
    };

    socket.onmessage = async (e) => {
      if (typeof e.data === 'string') {
        const msg = JSON.parse(e.data);
        // ElevenLabs streams MP3 chunks; decode once after all binary parts arrived.
        const parts = pendingAudioParts.current;
        pendingAudioParts.current = [];
        if (parts.length) {
          if (!audioCtx.current) {
            audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
          }
          try {
            const total = parts.reduce((sum, ab) => sum + ab.byteLength, 0);
            const merged = new Uint8Array(total);
            let offset = 0;
            for (const ab of parts) {
              merged.set(new Uint8Array(ab), offset);
              offset += ab.byteLength;
            }
            const buf = await audioCtx.current.decodeAudioData(
              merged.buffer.slice(merged.byteOffset, merged.byteOffset + merged.byteLength)
            );
            playBuffer(buf);
          } catch (err) {
            console.error("Audio decode error:", err);
          }
        }
        if (msg.error) {
          setStatus(msg.error); setStatusType("error");
        } else if (msg.query !== undefined) {
          setTranscript(msg.query ? `"${msg.query}"` : '');
          if (msg.response_text) setAiResponse(msg.response_text);
          if (msg.timings) {
            setLatency({
              stt: `${msg.timings.stt_ms?.toFixed(0)}ms`,
              vector: `${msg.timings.retrieval_ms?.toFixed(0)}ms`,
              ttfs: `${msg.timings.first_sentence_ms?.toFixed(0)}ms`,
              total: `${msg.timings.total_ms?.toFixed(0)}ms`,
            });
          }
          setStatus("Ready for next question."); setStatusType("success");
        }
      } else {
        pendingAudioParts.current.push(e.data);
      }
    };

    socket.onclose = () => {
      retryCount.current += 1;
      // Exponential backoff: 3s, 6s, 10s, 15s, max 30s
      const delay = Math.min(3000 * Math.min(retryCount.current, 3), 30000);
      setStatus(`Server waking up… retrying in ${Math.round(delay / 1000)}s`);
      setStatusType("error");
      retryTimer.current = setTimeout(connect, delay);
    };

    ws.current = socket;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  // Audio playback
  const playBuffer = (buffer) => {
    const src = audioCtx.current.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.current.destination);
    const now = audioCtx.current.currentTime;
    if (nextPlayTime.current < now) nextPlayTime.current = now;
    src.start(nextPlayTime.current);
    nextPlayTime.current += buffer.duration;
    activeSources.current += 1;
    setIsSpeaking(true); setStatus("Speaking…"); setStatusType("speaking");
    src.onended = () => {
      activeSources.current = Math.max(0, activeSources.current - 1);
      if (activeSources.current === 0) {
        setIsSpeaking(false); setStatus("Ready for next question."); setStatusType("success");
      }
    };
  };

  // Recording
  const startRecording = async () => {
    if (!docLoadedRef.current) return;
    if (ws.current?.readyState !== WebSocket.OPEN) {
      setStatus("Not connected. Wait for server to wake up."); setStatusType("error");
      return;
    }
    if (!audioCtx.current) {
      audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.current.state === 'suspended') await audioCtx.current.resume();

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) {
      recognition.current = new SR();
      recognition.current.continuous = true;
      recognition.current.interimResults = true;
      recognition.current.lang = 'en-US';
      recognition.current.onresult = (ev) => {
        let interim = '', final = '';
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const t = ev.results[i][0].transcript;
          ev.results[i].isFinal ? (final += t) : (interim += t);
        }
        setLiveTranscript(final || interim);
      };
      recognition.current.onerror = (ev) => console.warn("SR:", ev.error);
      try { recognition.current.start(); } catch (_) { }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];
      mediaRecorder.current.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunks.current.push(ev.data);
      };
      mediaRecorder.current.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(blob);
          setStatus("Processing…"); setStatusType("processing");
          setLatency({ stt: '--', vector: '--', ttfs: '--', total: '--' });
          setAiResponse("");
        }
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.current.start();
      setIsRecording(true);
      setLiveTranscript(""); setTranscript("");
      setStatus("Listening…"); setStatusType("recording");
    } catch (err) {
      console.error(err);
      setStatus("Microphone access denied."); setStatusType("error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
      mediaRecorder.current.stop();
    }
    setIsRecording(false);
    setLiveTranscript("");
    try { recognition.current?.stop(); } catch (_) { }
    recognition.current = null;
  };

  const handleMouseDown = (e) => {
    if (!docLoadedRef.current) return;
    const btn = micBtn.current;
    if (btn) {
      const rect = btn.getBoundingClientRect();
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      const size = Math.max(btn.offsetWidth, btn.offsetHeight);
      ripple.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - rect.left - size / 2}px;top:${e.clientY - rect.top - size / 2}px`;
      btn.appendChild(ripple);
      ripple.addEventListener('animationend', () => ripple.remove());
    }
    startRecording();
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus(`Uploading ${file.name}…`); setStatusType("processing");
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${apiOrigin()}/upload-document`, { method: "POST", body: fd });
      const data = await res.json();
      if (res.ok && data.status === "indexed") {
        setStatus(`Indexed ${data.chunk_count} chunks. Hold mic to speak.`);
        setStatusType("success");
        setDocumentLoaded(true);
        setUploadedFilename(file.name);
      } else {
        setStatus(data.detail || "Upload failed."); setStatusType("error");
      }
    } catch {
      setStatus(`Failed to upload ${file.name}.`); setStatusType("error");
    }
  };

  const latColor = (val, threshold) => {
    if (val === '--') return {};
    const n = parseFloat(val);
    return isNaN(n) ? {} : { color: n < threshold ? '#10b981' : '#ef4444' };
  };

  const displayText = isRecording && liveTranscript ? liveTranscript : transcript;

  return (
    <div className="app-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', margin: 0 }}>VoiceDoc AI</h1>
          <span className="label" style={{ fontSize: '0.7rem' }}>/ VOICE AGENT</span>
        </div>
        <div className="pill">Whisper + Claude + ElevenLabs</div>
      </header>

      <div className="content">
        <aside className="sidebar">
          <div>
            <div className="label" style={{ marginBottom: 16 }}>// DOCUMENT</div>
            <label className="dropzone">
              <input type="file" accept="application/pdf" style={{ display: 'none' }} onChange={handleUpload} />
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', color: 'var(--accent)' }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p>Click to upload PDF</p>
            </label>
            {documentLoaded && (
              <div style={{ marginTop: 16, textAlign: 'center' }}>
                <span className="pill" title={uploadedFilename}
                  style={{ display: 'inline-block', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  ✓ {uploadedFilename}
                </span>
              </div>
            )}
          </div>
          <div>
            <div className="label" style={{ marginBottom: 16 }}>// HOW IT WORKS</div>
            <div className="steps">
              {[
                'Upload a PDF to index in the vector database.',
                'Hold the mic button and speak your question.',
                'Claude retrieves context and speaks the answer.',
                'Monitor latency across the full pipeline.',
              ].map((text, i) => (
                <div className="step" key={i}>
                  <span className="step-num">{String(i + 1).padStart(2, '0')}</span>
                  <span className="step-text">{text}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="main-area">
          <div className="mic-container">
            <button
              ref={micBtn}
              className={`mic-button ${isRecording ? 'recording' : ''}`}
              disabled={!documentLoaded}
              onMouseDown={handleMouseDown}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
            >
              <svg className="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </button>

            {isSpeaking && (
              <div className="waveform">
                {[0.7, 0.9, 0.6, 1.0, 0.75].map((dur, i) => (
                  <div key={i} className="wave-bar" style={{ animationDuration: `${dur}s` }} />
                ))}
              </div>
            )}

            <div className="status-display">
              <div className={`status-text ${statusType}`}>{status}</div>
              {displayText && (
                <div className={`transcript ${isRecording && liveTranscript ? 'live' : ''}`}>
                  {displayText}
                </div>
              )}
              {aiResponse && (
                <div style={{ marginTop: 16, width: '100%', maxWidth: 480, textAlign: 'left' }}>
                  <div className="label" style={{ fontSize: '0.65rem', marginBottom: 6 }}>// AI RESPONSE</div>
                  <div className="ai-response">{aiResponse}</div>
                </div>
              )}
            </div>
          </div>

          <div className="latency-grid">
            {[
              { label: 'STT Recognition', val: latency.stt, threshold: 500 },
              { label: 'Vector Retrieval', val: latency.vector, threshold: 100 },
              { label: 'TTFS First Audio', val: latency.ttfs, threshold: 1000 },
              { label: 'Total Pipeline', val: latency.total, threshold: 2000 },
            ].map(({ label, val, threshold }) => (
              <div className="latency-card" key={label}>
                <div className="label">{label}</div>
                <div className="latency-val" style={latColor(val, threshold)}>{val}</div>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}