// App.jsx
// -------
// Root React component.
// Sets up routing between pages (ECG, EEG, Radar, Doppler).

import ECGPage from './pages/ECGPage.jsx'
function App() {
  return (
    <div className="App">
      <h1>Multi-Channel Signal Viewer</h1>
      <ECGPage />
    </div>
  )
}

export default App