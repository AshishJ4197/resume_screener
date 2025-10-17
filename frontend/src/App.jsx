// src/App.jsx
import { useMemo, useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function ScoreCircle({ score, threshold = 50 }) {
  const pct = Math.max(0, Math.min(100, Number(score || 0)));
  const radius = 52;
  const stroke = 10;
  const norm = radius - stroke / 2;
  const circ = 2 * Math.PI * norm;
  const dash = `${(pct / 100) * circ} ${circ - (pct / 100) * circ}`;
  const color = pct >= threshold ? '#16a34a' : '#dc2626'; // green/red

  return (
    <div className="d-flex flex-column align-items-center">
      <svg width="140" height="140" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={norm} stroke="#e5e7eb" strokeWidth={stroke} fill="none" />
        <circle
          cx="60"
          cy="60"
          r={norm}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={dash}
          transform="rotate(-90 60 60)"
        />
        <text x="60" y="66" textAnchor="middle" fontSize="28" fontWeight="600" fill="#111827">
          {pct}
        </text>
      </svg>
      <div className="small text-muted">Score / 100</div>
    </div>
  );
}

export default function App() {
  const [resumeFile, setResumeFile] = useState(null);

  const [jdMode, setJdMode] = useState('file'); // 'file' | 'text'
  const [jdFile, setJdFile] = useState(null);
  const [jdText, setJdText] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const apiUrl = useMemo(() => `${API_BASE}/api/v1/analyze`, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!resumeFile && !jdText) {
      setError('Please upload a Resume file and provide the JD (file or text).');
      return;
    }
    if (!resumeFile) {
      setError('Please upload a Resume file.');
      return;
    }
    if (jdMode === 'file' && !jdFile) {
      setError('Please upload a JD file or switch to “Paste Text”.');
      return;
    }
    if (jdMode === 'text' && !jdText.trim()) {
      setError('Please paste the JD text or switch to “Upload File”.');
      return;
    }

    try {
      setLoading(true);

      const fd = new FormData();
      // backend expects these field names
      fd.append('resume_file', resumeFile);
      if (jdMode === 'file') {
        fd.append('jd_file', jdFile);
      } else {
        fd.append('jd_text', jdText);
      }
      // you can pass optional options_json if you want to override anything
      // fd.append('options_json', JSON.stringify({ eligibility_threshold: 50 }));

      const resp = await axios.post(apiUrl, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(resp.data);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        'Something went wrong while analyzing. Check backend logs.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const eligColor = (result?.eligible ? 'success' : 'danger');

  return (
    <div className="container py-4">
      <header className="mb-4">
        <h1 className="display-6 fw-semibold text-center">Smart Resume Screener</h1>
        <p className="text-center text-muted mb-0">
          Upload a resume, provide the job description (file or text), then analyze.
        </p>
      </header>

      <div className="row g-4">
        {/* Form Card */}
        <div className="col-12 col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5 className="card-title mb-3">Inputs</h5>
              <form onSubmit={handleSubmit}>
                {/* Resume */}
                <div className="mb-3">
                  <label className="form-label">Resume (PDF/DOCX/TXT)</label>
                  <input
                    className="form-control"
                    type="file"
                    accept=".pdf,.docx,.txt"
                    onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
                  />
                  <div className="form-text">We only read the content to score against the JD.</div>
                </div>

                {/* JD mode */}
                <div className="row">
                  <div className="col-12 col-sm-4 mb-3">
                    <label className="form-label">JD Input</label>
                    <select
                      className="form-select"
                      value={jdMode}
                      onChange={(e) => setJdMode(e.target.value)}
                    >
                      <option value="file">Upload File</option>
                      <option value="text">Paste Text</option>
                    </select>
                  </div>
                  <div className="col-12 col-sm-8 mb-3">
                    {jdMode === 'file' ? (
                      <>
                        <label className="form-label">Job Description (PDF/DOCX/TXT)</label>
                        <input
                          className="form-control"
                          type="file"
                          accept=".pdf,.docx,.txt"
                          onChange={(e) => setJdFile(e.target.files?.[0] || null)}
                        />
                      </>
                    ) : (
                      <>
                        <label className="form-label">Job Description (Text)</label>
                        <textarea
                          className="form-control"
                          rows={6}
                          placeholder="Paste the JD text here…"
                          value={jdText}
                          onChange={(e) => setJdText(e.target.value)}
                        />
                      </>
                    )}
                  </div>
                </div>

                {error && (
                  <div className="alert alert-danger py-2">{error}</div>
                )}

                <div className="d-grid">
                  <button className="btn btn-primary" type="submit" disabled={loading}>
                    {loading ? 'Analyzing…' : 'Analyze'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>

        {/* Results Card */}
        <div className="col-12 col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5 className="card-title mb-3">Result</h5>

              {!result && !loading && (
                <div className="text-muted">Run an analysis to see the score and details here.</div>
              )}

              {loading && (
                <div className="d-flex align-items-center gap-2">
                  <div className="spinner-border text-primary" role="status" />
                  <span>Scoring your resume against the JD…</span>
                </div>
              )}

              {result && (
                <>
                  <div className="d-flex align-items-center gap-4 flex-wrap">
                    <ScoreCircle score={result.score} />
                    <div>
                      <div className="mb-2">
                        <span className={`badge bg-${eligColor} fs-6`}>
                          {result.eligible ? 'Eligible' : 'Not Eligible'}
                        </span>
                      </div>
                      <div className="small text-muted">
                        Run ID: <code>{result.run_id}</code>
                      </div>
                    </div>
                  </div>

                  {/* Contacts */}
                  <hr />
                  <h6 className="text-uppercase text-muted">Extracted Contacts</h6>
                  <div className="row g-2">
                    <div className="col-sm-6">
                      <div className="form-floating">
                        <input className="form-control" readOnly value={result.contacts?.name || ''} />
                        <label>Name</label>
                      </div>
                    </div>
                    <div className="col-sm-6">
                      <div className="form-floating">
                        <input className="form-control" readOnly value={result.contacts?.email || ''} />
                        <label>Email</label>
                      </div>
                    </div>
                    <div className="col-sm-6">
                      <div className="form-floating">
                        <input className="form-control" readOnly value={result.contacts?.phone || ''} />
                        <label>Phone</label>
                      </div>
                    </div>
                    <div className="col-sm-6">
                      <div className="form-floating">
                        <input className="form-control" readOnly value={result.contacts?.linkedin || ''} />
                        <label>LinkedIn</label>
                      </div>
                    </div>
                  </div>

                  {/* Narratives */}
                  <hr />
                  <div className="row g-3">
                    <div className="col-12">
                      <label className="form-label fw-semibold">What aligns well</label>
                      <textarea className="form-control" rows={4} readOnly value={result.narratives?.present_summary || ''} />
                    </div>
                    <div className="col-12">
                      <label className="form-label fw-semibold">Gaps & must-haves to address</label>
                      <textarea className="form-control" rows={4} readOnly value={result.narratives?.gaps_summary || ''} />
                    </div>
                    <div className="col-12">
                      <label className="form-label fw-semibold">Transferable strengths</label>
                      <textarea className="form-control" rows={4} readOnly value={result.narratives?.bonus_summary || ''} />
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <footer className="text-center text-muted small mt-4">
        <span>Backend: {API_BASE || 'http://localhost:8000'}</span>
      </footer>
    </div>
  );
}
