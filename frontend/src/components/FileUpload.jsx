import { useCallback, useRef, useState } from 'react';
import { analyzeContour } from '../services/api';

/**
 * The original contour-upload path, kept as a second way in.
 *
 * A surveyed 1 m contour map resolves a farm pond that a 30 m global elevation
 * model cannot, so this stays useful even though village search is the main
 * route now.
 */
export default function FileUpload({ onResult }) {
  const [file, setFile] = useState(null);
  const [resolution, setResolution] = useState(15);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const accept = (candidate) => {
    if (!candidate) return;
    const name = candidate.name.toLowerCase();
    if (!name.endsWith('.kml') && !name.endsWith('.kmz')) {
      setError('That file is not a KML or KMZ contour map.');
      return;
    }
    setError(null);
    setFile(candidate);
  };

  const submit = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setProgress(0);
    try {
      const data = await analyzeContour(file, resolution, 5, setProgress);
      onResult?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setProgress(0);
    }
  }, [file, resolution, onResult]);

  return (
    <div className="upload-block">
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files[0]);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".kml,.kmz"
          hidden
          onChange={(e) => accept(e.target.files[0])}
        />
        {file ? (
          <div className="selected-file">
            <strong>{file.name}</strong>
            <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span>
          </div>
        ) : (
          <>
            <p>Drop a KML or KMZ contour map</p>
            <span className="file-types">or click to choose one</span>
          </>
        )}
      </div>

      <label className="field-label" htmlFor="contour-res">
        Grid resolution: {resolution} m
      </label>
      <input
        id="contour-res"
        type="range"
        min={5}
        max={30}
        step={5}
        value={resolution}
        onChange={(e) => setResolution(Number(e.target.value))}
      />

      <button type="button" className="btn btn-ghost btn-full" onClick={submit} disabled={!file || busy}>
        {busy ? `Uploading… ${progress}%` : 'Analyse contour file'}
      </button>

      {error && <div className="error-banner">{error}</div>}
    </div>
  );
}
