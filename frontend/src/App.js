import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000/detect';

// Colours for the parking status (kept inline so no App.css change is needed)
const STATUS_STYLE = {
  ILLEGAL: { color: '#d32f2f', fontWeight: 'bold' },
  LEGAL: { color: '#2e7d32', fontWeight: 'bold' },
  UNKNOWN: { color: '#ed6c02', fontWeight: 'bold' },
};

const STATUS_LABEL = {
  ILLEGAL: 'ILLEGAL PARKING',
  LEGAL: 'Legal Parking',
  UNKNOWN: 'Unknown (no clear sign)',
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [vehicles, setVehicles] = useState([]);
  const [signs, setSigns] = useState([]);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [totalVehicles, setTotalVehicles] = useState(0);
  const [illegalCount, setIllegalCount] = useState(0);
  const [emailSent, setEmailSent] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setVehicles([]);
      setSigns([]);
      setProcessedImage(null);
      setTotalVehicles(0);
      setIllegalCount(0);
      setEmailSent(false);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Let axios/browser set the multipart boundary itself
      const response = await axios.post(API_URL, formData);
      const data = response.data;

      const vehicleList = Array.isArray(data.vehicles) ? data.vehicles : [];
      const signList = Array.isArray(data.parking_signs) ? data.parking_signs : [];

      setVehicles(vehicleList);
      setSigns(signList);
      setTotalVehicles(data.total_vehicles ?? vehicleList.length);
      setIllegalCount(
        data.illegal_vehicles ??
          vehicleList.filter((v) => v.parking_status === 'ILLEGAL').length
      );
      setEmailSent(Boolean(data.email_sent));

      // The backend sends raw base64, so it needs a data-URI prefix
      const img = data.image || data.result_image;
      setProcessedImage(img ? `data:image/jpeg;base64,${img}` : null);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === 'string'
          ? detail
          : err.message || 'Error processing image'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Vehicle Detection & Illegal Parking Detection</h1>
        <p>
          Upload an image to detect vehicles, extract license plates, and
          identify illegal parking
        </p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <input
            type="file"
            id="image-upload"
            accept="image/*"
            onChange={handleFileSelect}
            className="file-input"
          />
          <label htmlFor="image-upload" className="file-label">
            {selectedFile ? selectedFile.name : 'Choose an image'}
          </label>

          {preview && (
            <div className="preview-container">
              <h3>Original Image</h3>
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          <button
            onClick={handleDetect}
            disabled={!selectedFile || loading}
            className="detect-button"
          >
            {loading ? 'Processing...' : 'Detect Vehicles'}
          </button>

          {error && <p className="error-message">{error}</p>}
        </div>

        {processedImage && (
          <div className="results-section">
            <h3>Processed Image</h3>
            <img
              src={processedImage}
              alt="Processed"
              className="processed-image"
            />

            <div className="detections-list">
              <h3>Detection Results</h3>

              {vehicles.length === 0 && <p>No vehicles detected.</p>}

              {vehicles.map((vehicle, index) => {
                const status = vehicle.parking_status || 'UNKNOWN';
                const isIllegal = status === 'ILLEGAL';

                return (
                  <div
                    key={index}
                    className={`detection-card ${
                      isIllegal ? 'illegal-parking' : ''
                    }`}
                  >
                    <h4>Vehicle #{index + 1}</h4>

                    <p>
                      <strong>Type:</strong> {vehicle.vehicle_type}
                    </p>

                    <p>
                      <strong>Confidence:</strong>{' '}
                      {(vehicle.confidence * 100).toFixed(2)}%
                    </p>

                    <p>
                      <strong>License Plate:</strong>{' '}
                      {vehicle.license_plate || 'Not detected'}
                      {vehicle.license_plate && (
                        <>
                          {' '}
                          ({(vehicle.plate_confidence * 100).toFixed(0)}% OCR
                          {vehicle.plate_verified ? ', verified' : ', unverified'})
                        </>
                      )}
                    </p>

                    <p>
                      <strong>Parking Status:</strong>{' '}
                      <span style={STATUS_STYLE[status] || STATUS_STYLE.UNKNOWN}>
                        {STATUS_LABEL[status] || status}
                      </span>
                    </p>

                    <p>
                      <strong>Reason:</strong> {vehicle.parking_reason}
                    </p>

                    {/* associated_sign is an OBJECT - render its fields, never the object itself */}
                    {vehicle.associated_sign && (
                      <p>
                        <strong>Nearest Sign:</strong>{' '}
                        {vehicle.associated_sign.class_name} (
                        {(vehicle.associated_sign.confidence * 100).toFixed(0)}%)
                      </p>
                    )}
                  </div>
                );
              })}

              <div className="parking-summary">
                <p>
                  <strong>Total Vehicles Detected:</strong> {totalVehicles}
                </p>

                {/* signs is an ARRAY of objects - show the count, then each sign's fields */}
                <p>
                  <strong>Parking Signs Detected:</strong> {signs.length}
                </p>
                {signs.length > 0 && (
                  <ul>
                    {signs.map((sign, i) => (
                      <li key={i}>
                        {sign.class_name} – {(sign.confidence * 100).toFixed(0)}%
                      </li>
                    ))}
                  </ul>
                )}

                <p>
                  <strong>Illegal Vehicles:</strong> {illegalCount}
                </p>

                {emailSent && (
                  <p className="email-sent">📧 Email notification sent!</p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;