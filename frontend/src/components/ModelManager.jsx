import React, { useState, useEffect } from 'react';
import { Upload, Play, CheckCircle, AlertCircle } from 'lucide-react';

export default function ModelManager() {
    const [models, setModels] = useState([]);
    const [currentModel, setCurrentModel] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [training, setTraining] = useState(false);

    const fetchModels = () => {
        fetch('/api/models')
            .then(res => res.json())
            .then(data => {
                setModels(data.models);
                setCurrentModel(data.current_model);
            });
    };

    useEffect(() => {
        fetchModels();
        const interval = setInterval(fetchModels, 5000);
        return () => clearInterval(interval);
    }, []);

    const loadModel = (name) => {
        fetch(`/api/models/load?model_name=${name}`, { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') fetchModels();
            });
    };

    const handleFileUpload = async (e, type) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setUploading(true);
        try {
            let url = type === 'dataset' ? '/api/train' : '/api/models/upload';
            if (type === 'dataset') setTraining(true);

            const res = await fetch(url, { method: 'POST', body: formData });
            const data = await res.json();

            if (data.status === 'success') {
                alert(data.message);
                fetchModels();
            } else {
                alert('Error: ' + data.detail);
            }
        } catch (err) {
            alert('Upload failed');
        } finally {
            setUploading(false);
            if (type === 'dataset') setTraining(false); // In reality training is background, but we stop spinner
        }
    };

    return (
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
            <h2 className="text-2xl font-bold mb-4">Model Management</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <h3 className="text-xl font-semibold mb-2">Available Models</h3>
                    <ul className="space-y-2">
                        {models.map(model => (
                            <li key={model} className="flex items-center justify-between bg-gray-700 p-3 rounded">
                                <span className={currentModel === model ? 'text-green-400 font-bold' : 'text-gray-300'}>
                                    {model} {currentModel === model && '(Active)'}
                                </span>
                                {currentModel !== model && (
                                    <button
                                        onClick={() => loadModel(model)}
                                        className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-sm"
                                    >
                                        Load
                                    </button>
                                )}
                            </li>
                        ))}
                        {models.length === 0 && <li className="text-gray-500">No models found.</li>}
                    </ul>
                </div>

                <div className="space-y-4">
                    <div className="bg-gray-700 p-2 rounded">
                        <h3 className="font-semibold mb-2 flex items-center"><Upload className="w-4 h-4 mr-2" /> Upload Dataset (Train)</h3>
                        <p className="text-sm text-gray-400 mb-2">Upload a .zip file containing images to train a new model.</p>
                        <input
                            type="file"
                            accept=".zip"
                            onChange={(e) => handleFileUpload(e, 'dataset')}
                            disabled={uploading}
                            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                        />
                    </div>

                    <div className="bg-gray-700 p-2 rounded">
                        <h3 className="font-semibold mb-2 flex items-center"><Upload className="w-4 h-4 mr-2" /> Upload Pre-trained Model</h3>
                        <p className="text-sm text-gray-400 mb-2">Upload a .pth file directly.</p>
                        <input
                            type="file"
                            accept=".pth"
                            onChange={(e) => handleFileUpload(e, 'model')}
                            disabled={uploading}
                            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-600 file:text-white hover:file:bg-green-700"
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
