// ECGPage.jsx
import { useState, useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { fetchEcgCycles } from "../services/ecgService.jsx";

// Map index → ECG lead names
const ECG_LEAD_NAMES = [
  "I", "II", "III", "aVR", "aVL", "aVF",
  "V1", "V2", "V3", "V4", "V5", "V6"
];

function ECGPage() {
  const [recordNumber, setRecordNumber] = useState("98");
  const [leads, setLeads] = useState([0, 1, 2]); // Default first 3 leads
  const [cycles, setCycles] = useState([]);
  const [currentCycle, setCurrentCycle] = useState(0);
  const [fs, setFs] = useState(500);
  const timeoutRef = useRef(null);

  // Fetch ECG cycles
  useEffect(() => {
    async function loadCycles() {
      try {
        const data = await fetchEcgCycles(recordNumber, leads);
        setCycles(data.cycles);
        setFs(data.fs || 500);
        setCurrentCycle(0);
      } catch (error) {
        console.error("Failed to load ECG cycles:", error);
      }
    }
    loadCycles();
  }, [recordNumber, leads]);

  // Auto-loop with dynamic timing
  useEffect(() => {
    if (cycles.length === 0) return;

    const cycleData = cycles[currentCycle] || [];
    const cycleDuration = (cycleData.length / fs) * 1000; // duration in ms

    timeoutRef.current = setTimeout(() => {
      setCurrentCycle((prev) => (prev + 1) % cycles.length);
    }, cycleDuration);

    return () => clearTimeout(timeoutRef.current);
  }, [currentCycle, cycles, fs]);

  // Prepare plot data
  const cycleData = cycles[currentCycle] || [];

  const toggleLead = (leadIdx) => {
    if (leads.includes(leadIdx)) {
      setLeads(leads.filter((l) => l !== leadIdx));
    } else if (leads.length < 3) {
      setLeads([...leads, leadIdx]);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>ECG Viewer</h2>

      <div>
        <label>
          Record number:{" "}
          <input
            type="text"
            value={recordNumber}
            onChange={(e) => setRecordNumber(e.target.value)}
            style={{ width: "50px" }}
          />
        </label>
      </div>

      <div style={{ marginTop: "10px" }}>
        {Array.from({ length: 12 }, (_, i) => (
          <label key={i} style={{ marginRight: "10px" }}>
            <input
              type="checkbox"
              checked={leads.includes(i)}
              onChange={() => toggleLead(i)}
            />
            {ECG_LEAD_NAMES[i]}
          </label>
        ))}
      </div>

      <div style={{ marginTop: "20px" }}>
        {leads.map((leadIdx, i) => {
          const leadName = ECG_LEAD_NAMES[leadIdx];
          const trace = {
            x: cycleData.map((_, t) => t / fs),
            y: cycleData.map((row) => row[i]),
            type: "scatter",
            mode: "lines",
            name: leadName,
          };

          return (
            <Plot
              key={leadIdx}
              data={[trace]}
              layout={{
                width: 900,
                height: 250,
                title: `${leadName} (Cycle ${currentCycle + 1} / ${cycles.length})`,
                xaxis: { title: "Time (s)" },
                yaxis: { title: "Amplitude" },
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

export default ECGPage;
