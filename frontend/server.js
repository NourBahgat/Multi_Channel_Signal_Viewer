const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Set up multer for file upload
const upload = multer({ dest: 'uploads/' });

// Serve static frontend files (optional)
app.use(express.static('public'));

// Endpoint to handle file upload
app.post('/predict', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        // Send file to Flask API
        const formData = new FormData();
        const fileStream = fs.createReadStream(req.file.path);
        formData.append('file', fileStream);

        const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
            headers: formData.getHeaders()
        });

        // Delete temp file after sending
        fs.unlinkSync(req.file.path);

        res.json({ prediction: response.data.prediction });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Error connecting to Flask API' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Node server running at http://localhost:${PORT}`);
});
