import { useState, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Pipeline Stepper State (0: Idle, 1: Acquired, 2: Gating/Prep, 3: CNN inference, 4: Done)
  const [pipelineStep, setPipelineStep] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  // Add this right under your other useState declarations
  const [history, setHistory] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const recordedFile = new File([audioBlob], "live_signal.webm", { type: 'audio/webm' });
        setFile(recordedFile);
        setAudioUrl(URL.createObjectURL(recordedFile));
        stream.getTracks().forEach(track => track.stop());
        setPipelineStep(1); // Signal Acquired
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setResult(null);
      setHeatmap(null);
      setPipelineStep(0);
    } catch (err) {
      alert("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile)); // ADD THIS LINE
      setResult(null);
      setHeatmap(null);
      setPipelineStep(1);
    }
  };

  const simulatePipelineAndPredict = async () => {
    if (!file) return;
    setLoading(true);
    
    // Simulate pipeline visually before API finishes
    setPipelineStep(2); // Preprocessing
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Small delay to let the UI update step 2
      await new Promise(resolve => setTimeout(resolve, 800));
      setPipelineStep(3); // Feature Extraction & Inference

      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) throw new Error("Backend error");
      const data = await response.json();
      
      setResult(data);
      // Add this right after setResult(data);
      const newHistoryItem = {
        id: Date.now(),
        filename: file.name,
        language: data.predicted_language,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
      };
      setHistory(prev => [newHistoryItem, ...prev].slice(0, 5)); // Keeps only the last 5

      if (data.heatmap) setHeatmap(`data:image/png;base64,${data.heatmap}`);
      
      setPipelineStep(4); // Complete
    } catch (error) {
      console.error(error);
      alert("System failed to process the audio.");
      setPipelineStep(0);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Automatic Language Identification</h1>
        <p>CNN Architecture with Spectral Gating & Mel-Spectrogram Analysis</p>
      </header>

      <main className="dashboard-grid">
        
        {/* Left Column: Input & Pipeline */}
        <div className="left-column">
          <div className="card" style={{ marginBottom: '24px' }}>
            <h2 className="card-title">1. Audio Acquisition</h2>
            
            <div className="controls">
              <button className="btn" onClick={() => fileInputRef.current.click()}>
                Upload MP3/WAV
              </button>
              <input type="file" accept="audio/*" ref={fileInputRef} onChange={handleFileChange} hidden />
            </div>

            <button 
              className={`record-btn ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
            >
              {isRecording ? "Stop Recording" : "Start Live Mic"}
            </button>
            
            {file && !isRecording && (
              <p style={{marginTop: '15px', color: '#64748b', fontSize: '0.9rem'}}>
                File Loaded: <b>{file.name}</b>
              </p>
            )}

            {/* NEW AUDIO PLAYER */}
            {audioUrl && !isRecording && (
              <div style={{ marginTop: '15px', background: 'rgba(255,255,255,0.5)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(226, 232, 240, 0.8)' }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>PLAYBACK BUFFER</p>
                <audio controls src={audioUrl} style={{ width: '100%', height: '35px' }} />
              </div>
            )}

            <button 
              className="btn btn-primary" 
              style={{ width: '100%', marginTop: '20px' }} 
              onClick={simulatePipelineAndPredict} 
              disabled={loading || !file || isRecording}
            >
              {loading ? "Processing..." : "Run Analysis"}
            </button>
          </div>

          <div className="card">
            <h2 className="card-title">System Pipeline</h2>
            <div className="pipeline">
              <div className={`step ${pipelineStep >= 1 ? 'completed' : ''} ${pipelineStep === 1 ? 'active' : ''}`}>
                <div className="step-indicator"></div>
                Signal Acquired (Memory Buffer)
              </div>
              <div className={`step ${pipelineStep >= 2 ? 'completed' : ''} ${pipelineStep === 2 ? 'active' : ''}`}>
                <div className="step-indicator"></div>
                Spectral Gating & Silence Trimming
              </div>
              <div className={`step ${pipelineStep >= 3 ? 'completed' : ''} ${pipelineStep === 3 ? 'active' : ''}`}>
                <div className="step-indicator"></div>
                Mel-Spectrogram Generation & CNN Inference
              </div>
              <div className={`step ${pipelineStep === 4 ? 'completed active' : ''}`}>
                <div className="step-indicator"></div>
                Classification Complete
              </div>
            </div>
          </div>

          {/* COMPACT HISTORY LOG */}
          {history.length > 0 && (
            <div className="card" style={{ marginTop: '24px', padding: '16px' }}> {/* Reduced padding from 24px to 16px */}
              <h2 className="card-title" style={{ fontSize: '0.8rem', marginBottom: '12px', color: '#64748b' }}>Recent Analysis Log</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}> {/* Reduced gap between items */}
                {history.map((item) => (
                  <div key={item.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 12px', background: 'rgba(255,255,255,0.4)', borderRadius: '6px', border: '1px solid rgba(226, 232, 240, 0.5)' }}>
                    <div>
                      <span style={{ fontWeight: 600, color: '#1e293b', fontSize: '0.85rem' }}>{item.language}</span>
                      <span style={{ color: '#64748b', fontSize: '0.75rem', marginLeft: '6px' }}>({item.confidence})</span>
                    </div>
                    <span style={{ color: '#94a3b8', fontSize: '0.7rem', fontFamily: 'monospace' }}>{item.time}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Right Column: Visuals & Results */}
        <div className="right-column">
          
          <div className="card" style={{ marginBottom: '24px' }}>
            <h2 className="card-title">2. Acoustic Feature Map (Mel-Spectrogram)</h2>
            <div className="heatmap-container">
              {heatmap ? (
                <img src={heatmap} alt="Audio Spectrogram Heatmap" />
              ) : (
                <span style={{color: '#94a3b8', fontSize: '0.9rem'}}>Awaiting Signal Analysis...</span>
              )}
            </div>
          </div>

          <div className="card">
            <h2 className="card-title">3. Prediction Matrix</h2>
            
            {result ? (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ height: '180px', marginBottom: '20px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.distribution} layout="vertical" margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
                      <XAxis type="number" domain={[0, 100]} hide />
                      <YAxis dataKey="language" type="category" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12, fontWeight: 500}} width={80} interval={0} />
                      <Tooltip 
                        cursor={{fill: '#f1f5f9'}} 
                        contentStyle={{background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'}} 
                        itemStyle={{color: '#2563eb', fontWeight: 'bold'}}
                        formatter={(value) => [`${parseFloat(value).toFixed(2)}%`, 'Confidence']}
                      />
                      <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={16}>
                        {result.distribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.language.toUpperCase() === result.predicted_language.toUpperCase() ? '#2563eb' : '#cbd5e1'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                
                <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '20px' }}>
                  <p style={{margin: 0, color: '#64748b', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600}}>Final Classification</p>
                  <h3 className="result-text">{result.predicted_language}</h3>
                  <p className="confidence-text">Confidence Level: {result.confidence}</p>
                </div>
              </div>
            ) : (
              <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '0.9rem' }}>
                No probability data available.
              </div>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;