import React, { useState } from 'react';
import { Upload, AlertTriangle, CheckCircle, Loader2, ImageIcon, ScanSearch, RefreshCcw } from 'lucide-react';

export default function DetectionPage() {
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
            alert('检测失败，请重试');
        } finally {
            setLoading(false);
        }
    };

    const resetDetection = () => {
        setResult(null);
        setPreview(null);
    };

    return (
        <div className="p-4 max-w-6xl">
            {/* 页面标题 */}
            <div className="mb-8">
                <h1 className="text-[20px] font-bold text-black flex items-center">
                    <ScanSearch className="w-5 h-5 mr-2 text-black" /> 实时异常检测模块
                </h1>
                <p className="text-black mt-1 uppercase tracking-tight text-[14px]">Real-time Inference & Anomaly Visualization</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* 上传区域 */}
                <div className="column-card flex flex-col shadow-sm">
                    <div className="p-4 border-b border-slate-300 bg-slate-50/50 flex items-center justify-between">
                        <div>
                            <h2 className="text-[18px] font-bold text-black uppercase tracking-widest">原始图像采集</h2>
                            <p className="text-[14px] text-black mt-1 italic uppercase tracking-tighter font-medium">Original Image Acquisition</p>
                        </div>
                        {preview && (
                            <button
                                onClick={resetDetection}
                                className="text-[12px] bg-white border border-slate-300 px-2 py-1 rounded text-black hover:text-blue-600 hover:border-blue-300 transition-all font-bold uppercase tracking-widest flex items-center"
                            >
                                <RefreshCcw size={10} className="mr-1" /> 重新采集
                            </button>
                        )}
                    </div>

                    <div className="p-4">
                        <div className="border border-dashed border-slate-400 rounded-sm bg-slate-50 hover:bg-white transition-all relative aspect-square flex items-center justify-center overflow-hidden shadow-inner group">
                            <input
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />

                            {preview ? (
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="max-h-full max-w-full object-contain shadow-lg"
                                />
                            ) : (
                                <div className="flex flex-col items-center p-4 pointer-events-none group-hover:scale-110 transition-transform">
                                    <div className="w-16 h-16 bg-white rounded border border-slate-200 flex items-center justify-center mb-4 shadow-sm">
                                        <ImageIcon className="w-8 h-8 text-black" />
                                    </div>
                                    <p className="text-black font-bold text-[14px] uppercase tracking-widest">点击或拖拽图像</p>
                                    <p className="text-[12px] text-black mt-2 italic font-medium">支持 PNG / JPG 工控机实时抓取</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* 结果区域 */}
                <div className="column-card flex flex-col shadow-sm overflow-hidden">
                    <div className="p-4 border-b border-slate-300 bg-slate-50/50 flex items-center justify-between">
                        <div>
                            <h2 className="text-[18px] font-bold text-black uppercase tracking-widest">检测分析结果</h2>
                            <p className="text-[14px] text-black mt-1 italic uppercase tracking-tighter font-medium">Anomaly Analysis Report</p>
                        </div>
                        {/* 检测状态徽章 */}
                        {result && (
                            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-sm text-[12px] font-bold uppercase tracking-widest border shadow-sm ${result.is_anomaly
                                    ? 'bg-red-50 text-red-700 border-red-200'
                                    : 'bg-green-50 text-green-700 border-green-200'
                                }`}>
                                {result.is_anomaly ? (
                                    <AlertTriangle className="w-3.5 h-3.5" />
                                ) : (
                                    <CheckCircle className="w-3.5 h-3.5" />
                                )}
                                {result.is_anomaly ? '异常 (Anomaly)' : '正常 (Normal)'}
                            </div>
                        )}
                    </div>

                    <div className="p-4 flex-1 flex flex-col overflow-y-auto">
                        {loading ? (
                            <div className="flex-1 flex items-center justify-center">
                                <div className="text-center">
                                    <Loader2 className="w-10 h-10 text-blue-500 animate-spin mx-auto mb-4" />
                                    <p className="text-[14px] font-bold text-black tracking-widest uppercase animate-pulse">正在运行 PatchCore 算法...</p>
                                </div>
                            </div>
                        ) : result ? (
                            <div className="space-y-6 flex-1 flex flex-col">
                                {/* 分数信息 */}
                                <div className={`p-4 border-l-4 rounded-sm shadow-sm flex flex-col space-y-2 ${result.is_anomaly ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'
                                    }`}>
                                    <div className="flex justify-between items-end">
                                        <span className="text-[10px] font-extrabold uppercase tracking-widest opacity-60">Anomaly Score</span>
                                        <span className="text-xl font-mono font-black italic">{result.score.toFixed(4)}</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-200/50 rounded-full overflow-hidden">
                                        <div 
                                            className={`h-full transition-all duration-1000 ${result.is_anomaly ? 'bg-red-500' : 'bg-green-500'}`} 
                                            style={{ width: `${Math.min(100, (result.score / (result.threshold || 1)) * 50)}%` }}
                                        ></div>
                                    </div>
                                    {result.threshold && (
                                        <div className="flex justify-between text-[9px] font-bold text-slate-400 uppercase">
                                            <span>Threshold: {result.threshold.toFixed(4)}</span>
                                            <span>Score / Threshold: {(result.score / result.threshold * 100).toFixed(1)}%</span>
                                        </div>
                                    )}
                                </div>

                                {/* 热力图 */}
                                {result.heatmap ? (
                                    <div className="flex-1 flex flex-col">
                                        <label className="text-[10px] font-extrabold text-slate-500 uppercase mb-2 tracking-widest">Heatmap Localization</label>
                                        <div className="flex-1 aspect-square rounded-sm border border-slate-300 overflow-hidden bg-slate-900 shadow-lg relative">
                                            <img
                                                src={`data:image/png;base64,${result.heatmap}`}
                                                alt="Heatmap"
                                                className="w-full h-full object-contain"
                                            />
                                            <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-black/60 rounded text-[9px] text-white font-bold tracking-widest">HOTMAP_GEN</div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex-1 flex items-center justify-center border border-slate-200 border-dashed rounded bg-slate-50/50">
                                        <p className="text-[10px] text-slate-400 italic">No heatmap available for this result</p>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="flex-1 flex items-center justify-center">
                                <div className="text-center opacity-70">
                                    <div className="w-16 h-16 bg-slate-100 rounded border border-slate-200 flex items-center justify-center mx-auto mb-4">
                                        <ScanSearch className="w-8 h-8 text-black" />
                                    </div>
                                    <p className="text-black font-bold text-[14px] uppercase tracking-widest">等待数据分析</p>
                                    <p className="text-[12px] text-black mt-2 italic font-medium">请从左侧面板采集新图像</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
