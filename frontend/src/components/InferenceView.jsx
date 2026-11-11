import React, { useState } from 'react';
import { Upload, AlertTriangle, CheckCircle } from 'lucide-react';

export default function InferenceView() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [preview, setPreview] = useState(null);

    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Preview
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result);
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        setLoading(true);
        setResult(null);

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok) {
                setResult(data);
            } else {
                alert('Error: ' + data.detail);
            }
        } catch (err) {
            alert('Inference failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4">Real-time Inference</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                    <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors relative">
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageUpload}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        {!preview ? (
                            <div className="flex flex-col items-center">
                                <Upload className="w-12 h-12 text-gray-400 mb-2" />
                                <p className="text-gray-400">Click or drag image to analyze</p>
                            </div>
                        ) : (
                            <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded" />
                        )}
                    </div>
                </div>

                <div>
                    {loading && (
                        <div className="flex items-center justify-center h-full">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                        </div>
                    )}

                    {result && (
                        <div className="space-y-4">
                            <div className={`p-2 rounded-lg ${result.score > 0.5 ? 'bg-red-900/50 border border-red-500' : 'bg-green-900/50 border border-green-500'}`}>
                                <div className="flex items-center mb-2">
                                    {result.score > 0.5 ? <AlertTriangle className="text-red-500 mr-2" /> : <CheckCircle className="text-green-500 mr-2" />}
                                    <h3 className="text-xl font-bold">{result.score > 0.5 ? 'Anomaly Detected' : 'Normal'}</h3>
                                </div>
                                <p className="text-gray-300">Anomaly Score: <span className="font-mono font-bold">{result.score.toFixed(4)}</span></p>
                            </div>

                            <div>
                                <h4 className="font-semibold mb-2">Anomaly Heatmap</h4>
                                <img src={`data:image/png;base64,${result.heatmap}`} alt="Heatmap" className="w-full rounded border border-gray-700" />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
