import React, { useState, useEffect } from 'react';
import { Activity, Server, Database, Cpu } from 'lucide-react';

export default function Dashboard() {
    const [status, setStatus] = useState('Active');
    const [systemStats, setSystemStats] = useState({
        cpu: '12%',
        memory: '3.4GB',
        gpu: 'RTX 3080'
    });

    useEffect(() => {
        // In a real app, we would fetch system stats here
        const interval = setInterval(() => {
            // Simulate changing stats
            setSystemStats(prev => ({
                ...prev,
                cpu: Math.floor(Math.random() * 20 + 5) + '%'
            }));
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-800 p-6 rounded-lg shadow-lg flex items-center border border-gray-700">
                <div className="p-3 bg-blue-900/50 rounded-full mr-4">
                    <Activity className="text-blue-400 w-8 h-8" />
                </div>
                <div>
                    <h3 className="text-gray-400 text-sm font-medium">System Status</h3>
                    <span className="text-xl font-bold text-green-400">{status}</span>
                </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg shadow-lg flex items-center border border-gray-700">
                <div className="p-3 bg-purple-900/50 rounded-full mr-4">
                    <Cpu className="text-purple-400 w-8 h-8" />
                </div>
                <div>
                    <h3 className="text-gray-400 text-sm font-medium">CPU Usage</h3>
                    <span className="text-xl font-bold text-gray-200">{systemStats.cpu}</span>
                </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg shadow-lg flex items-center border border-gray-700">
                <div className="p-3 bg-yellow-900/50 rounded-full mr-4">
                    <Database className="text-yellow-400 w-8 h-8" />
                </div>
                <div>
                    <h3 className="text-gray-400 text-sm font-medium">Memory</h3>
                    <span className="text-xl font-bold text-gray-200">{systemStats.memory}</span>
                </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-lg shadow-lg flex items-center border border-gray-700">
                <div className="p-3 bg-green-900/50 rounded-full mr-4">
                    <Server className="text-green-400 w-8 h-8" />
                </div>
                <div>
                    <h3 className="text-gray-400 text-sm font-medium">GPU</h3>
                    <span className="text-xl font-bold text-gray-200">{systemStats.gpu}</span>
                </div>
            </div>
        </div>
    );
}
