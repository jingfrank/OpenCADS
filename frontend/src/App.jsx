import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import DatasetsPage from './pages/DatasetsPage';
import TrainingPage from './pages/TrainingPage';
import ModelsPage from './pages/ModelsPage';
import DetectionPage from './pages/DetectionPage';
import RealTimeDetectionPage from './pages/RealTimeDetectionPage';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Layout />}>
                    <Route index element={<Navigate to="/datasets" replace />} />
                    <Route path="datasets" element={<DatasetsPage />} />
                    <Route path="training" element={<TrainingPage />} />
                    <Route path="models" element={<ModelsPage />} />
                    <Route path="detection" element={<DetectionPage />} />
                    <Route path="realtime-detection" element={<RealTimeDetectionPage />} />
                </Route>
            </Routes>
        </BrowserRouter>
    );
}

export default App;
