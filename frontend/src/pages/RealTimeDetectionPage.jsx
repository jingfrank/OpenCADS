import React, { useState, useEffect } from 'react';
import {
    Database, Search,
    Play, Square, ChevronRight, CheckCircle2,
    AlertCircle, Zap,
    RefreshCw, Save, Send, Monitor, Upload,
    MoreHorizontal
} from 'lucide-react';

export default function RealTimeDetectionPage() {
    const [isDetecting, setIsDetecting] = useState(false);
    const [currentStatus, setCurrentStatus] = useState('ng'); // 'ok', 'ng'
    
    // 质检复核与历史回看模式
    const [viewMode, setViewMode] = useState('live'); // 'live', 'history'
    const [selectedHistory, setSelectedHistory] = useState(null);
    const [qcForm, setQcForm] = useState({ result: 'ok', notes: '', reviewer: '' });
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [datasets, setDatasets] = useState([]);
    const [selectedDataset, setSelectedDataset] = useState('');
    const [pendingImages, setPendingImages] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(-1);
    const [predictionData, setPredictionData] = useState(null);
    const [detectionHistory, setDetectionHistory] = useState([]);

    const [currentModel, setCurrentModel] = useState('');
    const [modelLoading, setModelLoading] = useState(false);

    // 获取真实模型列表和数据集列表
    useEffect(() => {
        const initData = async () => {
            try {
                // 模型
                const mRes = await fetch('/api/models');
                const mData = await mRes.json();
                setModels(mData.models || []);
                setCurrentModel(mData.current_model || '');
                if (mData.current_model) {
                    setSelectedModel(mData.current_model);
                } else if (mData.models?.length > 0) {
                    setSelectedModel(mData.models[0]);
                }

                // 数据集
                const dRes = await fetch('/api/datasets');
                const dData = await dRes.json();
                setDatasets(dData.datasets || []);
            } catch (err) {
                console.error('Initialization failed:', err);
            }
        };
        initData();
    }, []);

    // 自动加载选中的模型
    useEffect(() => {
        if (!selectedModel || selectedModel === currentModel) return;
        const loadModel = async () => {
            setModelLoading(true);
            try {
                const res = await fetch(`/api/models/load?model_name=${encodeURIComponent(selectedModel)}`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    setCurrentModel(selectedModel);
                } else {
                    alert('模型加载失败: ' + (data.detail || '未知错误'));
                }
            } catch (err) {
                alert('模型加载失败');
            } finally {
                setModelLoading(false);
            }
        };
        loadModel();
    }, [selectedModel]);

    // 监听数据集切换，获取待检测（未标记）图像
    useEffect(() => {
        if (!selectedDataset) return;
        const fetchPending = async () => {
            try {
                const res = await fetch(`/api/datasets/${encodeURIComponent(selectedDataset)}/images`);
                const data = await res.json();
                const unlabeled = (data.images || []).filter(img => !img.labeled);
                setPendingImages(unlabeled);
                setCurrentIndex(-1);
            } catch (err) {
                console.error('Failed to fetch dataset images:', err);
            }
        };
        fetchPending();
    }, [selectedDataset]);

    // 加载检测历史
    const fetchHistory = async () => {
        try {
            const res = await fetch('/api/detection/history?limit=50');
            const data = await res.json();
            setDetectionHistory((data.history || []).map(h => ({
                id: h.id,
                name: h.filename,
                dataset_name: h.dataset_name || '',
                time: h.timestamp,
                result: h.is_anomaly ? 'NG' : 'OK',
                score: typeof h.score === 'number' ? h.score.toFixed(3) : '—',
                qc_status: h.qc_status || 'pending',
                qc_result: h.qc_result || null,
                qc_notes: h.qc_notes || '',
                qc_reviewer: h.qc_reviewer || '',
                qc_timestamp: h.qc_timestamp || ''
            })));
        } catch (err) {
            console.error('Failed to fetch detection history:', err);
        }
    };

    useEffect(() => { fetchHistory(); }, []);

    // 核心检测循环
    useEffect(() => {
        let timer;
        if (isDetecting && pendingImages.length > 0 && currentIndex < pendingImages.length - 1) {
            timer = setTimeout(async () => {
                const nextIndex = currentIndex + 1;
                setCurrentIndex(nextIndex);
                const img = pendingImages[nextIndex];

                try {
                    // 1. 获取图像 Blob
                    const imgUrl = `/api/datasets/${encodeURIComponent(selectedDataset)}/images/${encodeURIComponent(img.filename)}`;
                    const blobRes = await fetch(imgUrl);
                    const blob = await blobRes.blob();

                    // 2. 发送预测请求（携带数据集名）
                    const formData = new FormData();
                    formData.append('file', blob, img.filename);
                    if (selectedDataset) formData.append('dataset_name', selectedDataset);
                    const predRes = await fetch('/api/predict', { method: 'POST', body: formData });
                    const res = await predRes.json();

                    // 3. 更新 UI
                    setPredictionData(res);
                    setCurrentStatus(res.is_anomaly ? 'ng' : 'ok');

                    // 4. 刷新历史记录（从后端获取）
                    fetchHistory();

                } catch (err) {
                    console.error('Detection failed:', err);
                    setIsDetecting(false);
                }
            }, 2000); // 模拟芯片检测节拍
        } else if (isDetecting && currentIndex >= pendingImages.length - 1 && pendingImages.length > 0) {
            setIsDetecting(false);
            alert('当前数据集未标记图像检测完成');
        }
        return () => clearTimeout(timer);
    }, [isDetecting, currentIndex, pendingImages, selectedDataset]);

    const handleHistoryClick = (item) => {
        setSelectedHistory(item);
        setViewMode('history');
        setQcForm({
            result: item.qc_result || 'ok',
            notes: item.qc_notes || '',
            reviewer: item.qc_reviewer || ''
        });
    };

    const handleQcSubmit = async () => {
        if (viewMode !== 'history' || !selectedHistory) {
            alert('请先在左侧选择要复核的历史记录');
            return;
        }
        if (!qcForm.reviewer.trim()) {
            alert('请填写复核人姓名');
            return;
        }
        try {
            const res = await fetch(`/api/detection/history/${selectedHistory.id}/qc`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(qcForm)
            });
            const data = await res.json();
            if (data.status === 'success') {
                // Update the item in-place in history list
                setDetectionHistory(prev => prev.map(h =>
                    h.id === selectedHistory.id
                        ? { ...h, qc_status: 'reviewed', qc_result: qcForm.result, qc_notes: qcForm.notes, qc_reviewer: qcForm.reviewer }
                        : h
                ));
                setSelectedHistory(prev => ({ ...prev, qc_status: 'reviewed', qc_result: qcForm.result, qc_notes: qcForm.notes, qc_reviewer: qcForm.reviewer }));
                // Brief confirmation then stay in history view
                alert(`复核已提交——判定：${qcForm.result.toUpperCase()}，复核人：${qcForm.reviewer}`);
            } else {
                alert('提交失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            alert('提交复核失败');
        }
    };

    const displayIsHistory = viewMode === 'history' && selectedHistory;
    const displayStatus = displayIsHistory ? (selectedHistory.result === 'NG' ? 'ng' : 'ok') : currentStatus;
    const displayScore = displayIsHistory ? selectedHistory.score : (predictionData?.score ? predictionData.score.toFixed(3) : '0.000');

    const pageStyles = `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.2); }
        .vertical-text { writing-mode: vertical-lr; transform: rotate(180deg); }
    `;

    return (
        <div className="flex flex-col h-full overflow-hidden bg-slate-100/30">
            {/* 顶层面包屑导航 */}
            <nav className="px-4 py-2 text-[20px] flex items-center bg-transparent border-b border-slate-300">
                <span className="text-blue-600 font-semibold cursor-pointer hover:underline">实时检测</span>
                <ChevronRight size={13} className="mx-1.5 text-slate-400" />
                <span className="text-black font-bold text-[20px]">实时检测{currentModel ? ` — ${currentModel}` : ''}</span>
            </nav>

            {/* 三栏主布局 */}
            <div className="flex-1 p-3 pt-1 flex space-x-3 overflow-hidden">

                {/* 第一栏：状态与历史 (左) - 280px */}
                <div className="w-[280px] flex-shrink-0 flex flex-col space-y-3 min-h-0">
                    {/* 数据源与模型卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            实时检测状态 & 数据源
                        </h3>
                        <div className="space-y-3">
                            <div>
                                <label className="block text-[14px] font-bold text-black mb-1">当前图像源 (数据集)</label>
                                <select
                                    className="w-full h-8 text-[14px] border border-slate-300 rounded bg-white text-black px-2 outline-none font-medium"
                                    value={selectedDataset}
                                    onChange={(e) => setSelectedDataset(e.target.value)}
                                >
                                    <option value="">请选择数据集...</option>
                                    {datasets.map(d => (
                                        <option key={d.name} value={d.name}>{d.name} ({d.image_count})</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-[14px] font-bold text-black mb-1 flex justify-between">
                                    检测模型 <span className="text-blue-600 cursor-pointer hover:underline">本地文件</span>
                                </label>
                                <select
                                    className="w-full h-8 text-[14px] border border-slate-300 rounded bg-white text-black px-2 outline-none font-bold"
                                    value={selectedModel}
                                    onChange={(e) => setSelectedModel(e.target.value)}
                                >
                                    {models.length === 0 ? (
                                        <option value="">暂无可用模型</option>
                                    ) : (
                                        models.map(m => (
                                            <option key={m} value={m}>{m}</option>
                                        ))
                                    )}
                                </select>
                                {modelLoading && (
                                    <div className="text-[12px] text-blue-600 font-bold mt-1 animate-pulse">模型加载中...</div>
                                )}
                                {!modelLoading && currentModel && (
                                    <div className="text-[12px] text-green-600 font-bold mt-1">✓ 已加载: {currentModel}</div>
                                )}
                            </div>
                            <div className="col-span-2 text-blue-800 text-[13px]">待检测数量: <span className="font-bold">{pendingImages.length}</span></div>
                            <div className="col-span-2 text-[13px]">当前进度: <span className="text-slate-500">{currentIndex + 1} / {pendingImages.length}</span></div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 pt-1">
                            <button
                                onClick={() => setIsDetecting(true)}
                                className={`h-9 flex items-center justify-center rounded font-bold text-[13px] shadow-sm transition-all ${isDetecting ? 'bg-slate-200 text-slate-500' : 'bg-green-600 text-white hover:bg-green-700'}`}
                            >
                                <Play className="w-3.5 h-3.5 mr-1.5" /> 启动实时检测
                            </button>
                            <button
                                onClick={() => setIsDetecting(false)}
                                className="h-9 bg-red-600 text-white hover:bg-red-700 flex items-center justify-center rounded font-bold text-[13px] shadow-sm"
                            >
                                <Square className="w-3.5 h-3.5 mr-1.5 fill-current" /> 停止检测
                            </button>
                        </div>
                    </div>

                    {/* 统计与历史卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300 flex-1 flex flex-col min-h-0 overflow-hidden">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            检测统计 & 历史记录
                        </h3>

                        {/* 比例条 */}
                        <div className="mb-4">
                            <div className="flex justify-between items-center mb-1.5">
                                <span className="text-[14px] font-bold text-black">今日 OK / NG 比例</span>
                            </div>
                            <div className="w-full h-4 bg-red-500 rounded-full flex overflow-hidden shadow-inner">
                                <div className="h-full bg-green-500" style={{ width: detectionHistory.length > 0 ? `${(detectionHistory.filter(h => h.result === 'OK').length / detectionHistory.length * 100).toFixed(1)}%` : '0%' }}></div>
                            </div>
                            <div className="mt-2 space-y-1">
                                <div className="flex justify-between text-[12px] font-bold">
                                    <span className="text-green-700">OK</span>
                                    <span>{detectionHistory.filter(h => h.result === 'OK').length}</span>
                                </div>
                                <div className="flex justify-between text-[12px] font-bold">
                                    <span className="text-red-700">NG</span>
                                    <span>{detectionHistory.filter(h => h.result === 'NG').length}</span>
                                </div>
                            </div>
                        </div>

                        {/* 历史列表 */}
                        <div className="flex-1 flex flex-col min-h-0">
                            <h4 className="text-[13px] font-bold text-black mb-2 flex items-center">
                                最近检测列表
                            </h4>
                            <div className="flex-1 overflow-y-auto pr-1 space-y-2 custom-scrollbar">
                                {detectionHistory.map(item => (
                                    <div key={item.id} onClick={() => handleHistoryClick(item)} className={`p-2 rounded border transition-colors cursor-pointer group ${selectedHistory?.id === item.id ? 'bg-blue-50 border-blue-400' : 'bg-white border-slate-200 hover:border-blue-300'}`}>
                                        <div className="flex justify-between items-center mb-1">
                                            <span className={`text-[13px] font-bold ${selectedHistory?.id === item.id ? 'text-blue-800' : 'text-black group-hover:text-blue-700'}`}>{item.name}</span>
                                            <span className={`text-[11px] px-1.5 py-0.5 rounded font-black ${item.qc_status === 'reviewed' ? 'bg-blue-100 text-blue-700' : (item.result === 'NG' ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600')}`}>
                                                {item.result} {item.qc_status === 'reviewed' && '☑️'}
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-[11px] font-bold text-slate-400">
                                            <span>{item.time}</span>
                                            <span>得分: {item.score}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* 第二栏：大屏可视化 (中) - flex-1 */}
                <div className="flex-1 flex flex-col space-y-3 min-w-0 overflow-hidden">
                    {/* 判定主面板 */}
                    <div className="column-card p-4 shadow-sm border border-slate-300 flex flex-col min-h-0 overflow-hidden flex-[1.5]">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            实时图像采集 & 缺陷可视化监测
                            {viewMode === 'history' && <span className="ml-auto bg-amber-200 text-amber-800 text-[12px] px-2 py-0.5 rounded font-bold shadow-sm">历史查看模式 (实时检测后台运行中)</span>}
                        </h3>

                        <div className="flex-1 flex flex-col min-h-0">
                            {/* 顶部判定大横条 */}
                            <div className="grid grid-cols-2 gap-4 mb-4 items-stretch">
                                <div className="space-y-2">
                                    <div className="text-[14px] font-bold text-black">判定结果 ({displayIsHistory ? '当时模型结果' : '模型最终结果'})</div>
                                    <div className={`h-20 rounded shadow-md flex items-center justify-center text-[42px] font-black italic tracking-tighter ${displayStatus === 'ng' ? 'bg-red-600 text-white' : 'bg-green-500 text-white'}`}>
                                        {displayStatus === 'ng' ? '不合格 NG' : '合格 OK'} {displayIsHistory ? '(历史)' : '(当前)'}
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="text-[14px] font-bold text-black">质检员复核结果</div>
                                    <div className={`h-20 rounded border-2 flex items-center justify-center text-[42px] font-black tracking-tighter italic ${displayIsHistory && selectedHistory.qc_status === 'reviewed' ? (selectedHistory.qc_result === 'ng' ? 'bg-red-100 border-red-400 text-red-600' : 'bg-green-100 border-green-400 text-green-600') : 'bg-slate-100 border-slate-300 text-slate-400'}`}>
                                        {displayIsHistory && selectedHistory.qc_status === 'reviewed' ? (selectedHistory.qc_result === 'ng' ? '复核 NG' : '复核 OK') : '待 复 核'}
                                    </div>
                                </div>
                            </div>

                            {/* 三图横向对比 */}
                            <div className="flex-1 grid grid-cols-3 gap-3 min-h-0 mb-4">
                                <div className="flex flex-col min-h-0">
                                    <h5 className="text-[13px] font-bold text-black mb-1.5 text-center uppercase">原始图像</h5>
                                    <div className="flex-1 bg-slate-900 rounded border border-slate-300 overflow-hidden flex items-center justify-center group relative cursor-zoom-in min-h-0">
                                        {displayIsHistory ? (
                                            <img
                                                src={selectedHistory.dataset_name
                                                    ? `/api/datasets/${encodeURIComponent(selectedHistory.dataset_name)}/images/${encodeURIComponent(selectedHistory.name)}`
                                                    : (selectedDataset ? `/api/datasets/${encodeURIComponent(selectedDataset)}/images/${encodeURIComponent(selectedHistory.name)}` : '')}
                                                alt="Origin"
                                                className="w-full h-full object-cover"
                                            />
                                        ) : currentIndex >= 0 ? (
                                            <img
                                                src={`/api/datasets/${encodeURIComponent(selectedDataset)}/images/${encodeURIComponent(pendingImages[currentIndex].filename)}`}
                                                alt="Origin"
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <div className="text-slate-500 text-[13px]">等待检测启动...</div>
                                        )}
                                        {(displayIsHistory || currentIndex >= 0) && <div className="absolute top-2 left-2 px-2 py-1 bg-black/50 text-white text-[11px] font-mono rounded">{displayIsHistory ? selectedHistory.name : pendingImages[currentIndex].filename}</div>}
                                        <div className="absolute bottom-2 left-2 text-white text-[11px] font-bold drop-shadow-sm">{displayIsHistory ? selectedHistory.time : new Date().toLocaleTimeString()}</div>
                                    </div>
                                </div>
                                <div className="flex flex-col min-h-0">
                                    <h5 className="text-[13px] font-bold text-black mb-1.5 text-center uppercase">模型热力图 (Heatmap)</h5>
                                    <div className="flex-1 bg-slate-900 rounded border border-slate-300 overflow-hidden flex items-center justify-center relative min-h-0">
                                        {displayIsHistory ? (
                                            <img src={`/api/detection/history/${selectedHistory.id}/heatmap`} alt="Heatmap" className="w-full h-full object-cover" />
                                        ) : predictionData?.heatmap ? (
                                            <img src={`data:image/png;base64,${predictionData.heatmap}`} alt="Heatmap" className="w-full h-full object-cover" />
                                        ) : (
                                            <div className="text-slate-700 text-[13px]">等待预测数据...</div>
                                        )}
                                        {/* 色带演示 */}
                                        <div className="absolute right-2 top-[10%] bottom-[10%] w-3 bg-gradient-to-t from-blue-700 via-yellow-400 to-red-600 rounded-full border border-white/20"></div>
                                        <div className="absolute right-6 top-[10%] text-white text-[11px] font-bold">1.0</div>
                                        <div className="absolute right-6 bottom-[10%] text-white text-[11px] font-bold">0.0</div>
                                    </div>
                                </div>
                                <div className="flex flex-col min-h-0">
                                    <h5 className="text-[13px] font-bold text-black mb-1.5 text-center uppercase">缺陷定位 & 二值图</h5>
                                    <div className="flex-1 bg-slate-900 rounded border border-slate-300 overflow-hidden flex items-center justify-center relative min-h-0">
                                        {displayIsHistory ? (
                                            <img src={`/api/detection/history/${selectedHistory.id}/mask`} alt="Binary" className="w-full h-full object-cover grayscale brightness-200 contrast-[500%]" />
                                        ) : predictionData?.mask ? (
                                            <img src={`data:image/png;base64,${predictionData.mask}`} alt="Binary" className="w-full h-full object-cover grayscale brightness-200 contrast-[500%]" />
                                        ) : (
                                            <div className="text-slate-700 text-[13px]">等待掩膜数据...</div>
                                        )}
                                        <div className="absolute inset-0 border-2 border-red-500/50 pointer-events-none"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 阈值分布图卡片 */}
                    <div className="column-card h-48 p-3 shadow-sm border border-slate-300 flex flex-col overflow-hidden">
                        <h3 className="text-[16px] font-extrabold text-black mb-2 flex items-center border-b border-slate-200 pb-1 uppercase tracking-tight">
                            检测阈值分布图 (当前 Batch)
                        </h3>
                        <div className="flex-1 bg-slate-50 flex items-end justify-between px-6 pb-2 pt-4 relative">
                            {/* 模拟坐标轴 */}
                            <div className="absolute left-4 top-4 bottom-4 w-px bg-slate-300"></div>
                            <div className="absolute left-4 bottom-2 right-4 h-px bg-slate-300"></div>

                            {/* 条形模拟 */}
                            {[20, 35, 45, 80, 100, 150, 180, 500, 160, 40].map((h, i) => (
                                <div key={i} className="w-6 bg-blue-300/40 border-t border-blue-400" style={{ height: `${h / 6}%` }}></div>
                            ))}

                            <span className="absolute bottom-1 right-6 text-[11px] font-bold text-slate-400">Score Value</span>
                            <span className="absolute left-6 top-3 text-[11px] font-bold text-slate-400 uppercase vertical-text">Quantity</span>
                        </div>
                    </div>
                </div>

                {/* 第三栏：详细数据与复核 (右) - 300px */}
                <div className="w-[300px] flex-shrink-0 flex flex-col space-y-3 min-h-0">
                    {/* 判定详细卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            质检复核 & 操作控制
                        </h3>
                        <div className="space-y-4">
                            <div>
                                <h4 className="text-[13px] font-bold text-black mb-1.5 opacity-60">当前判定详细信息</h4>
                                <div className="space-y-1.5 text-[14px] font-bold text-black">
                                    <div className="flex justify-between">
                                        <span>模型异常分数:</span>
                                        <span className={displayStatus === 'ng' ? 'text-red-700' : 'text-green-700'}>
                                            {displayScore}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>初步分类结果:</span>
                                        <div className="text-right">
                                            <div className={displayStatus === 'ng' ? 'text-red-600 font-bold' : 'text-green-600 font-bold'}>
                                                {displayStatus === 'ng' ? 'NG异常' : 'OK正常'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="p-3 bg-slate-50 rounded border border-slate-200 space-y-3">
                                <h4 className="text-[13px] font-bold text-black flex items-center">
                                    质检复核 (QC Review)
                                    <span className="ml-auto text-[10px] bg-slate-200 text-slate-500 px-1 rounded uppercase font-black">Key</span>
                                </h4>
                                <div>
                                    <label className="block text-[12px] font-bold text-black mb-1">复核判定结果</label>
                                    <select 
                                        className="w-full h-8 text-[13px] border border-slate-300 rounded bg-white text-black px-2 font-bold outline-none"
                                        value={qcForm.result}
                                        onChange={(e) => setQcForm(prev => ({ ...prev, result: e.target.value }))}
                                        disabled={!displayIsHistory}
                                    >
                                        <option value="ok">合格 OK</option>
                                        <option value="ng">不合格 NG</option>
                                        <option value="pending">待定</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-[12px] font-bold text-black mb-1">复核备注</label>
                                    <textarea
                                        className="w-full h-16 border border-slate-300 rounded p-2 text-[13px] text-black bg-white outline-none"
                                        placeholder="复核备注"
                                        value={qcForm.notes}
                                        onChange={(e) => setQcForm(prev => ({ ...prev, notes: e.target.value }))}
                                        disabled={!displayIsHistory}
                                    />
                                </div>
                                <div>
                                    <label className="block text-[12px] font-bold text-black mb-1">复核人</label>
                                    <input
                                        type="text"
                                        className="w-full h-8 text-[13px] border border-slate-300 rounded bg-white text-black px-2 outline-none"
                                        placeholder="请输入复核人姓名"
                                        value={qcForm.reviewer}
                                        onChange={(e) => setQcForm(prev => ({ ...prev, reviewer: e.target.value }))}
                                        disabled={!displayIsHistory}
                                    />
                                </div>
                                <button 
                                    onClick={handleQcSubmit}
                                    disabled={!displayIsHistory}
                                    className="w-full h-9 bg-blue-700 text-white rounded font-bold text-[13px] flex items-center justify-center shadow-md hover:bg-blue-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                                    <Send className="w-3.5 h-3.5 mr-2" /> 提交复核结果
                                </button>
                                {displayIsHistory && selectedHistory?.qc_status === 'reviewed' && (
                                    <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-[12px] space-y-0.5">
                                        <div className="font-extrabold text-blue-700 flex items-center gap-1">
                                            ☑ 已复核 — <span className={qcForm.result === 'ng' ? 'text-red-600' : qcForm.result === 'pending' ? 'text-amber-600' : 'text-green-700'}>{qcForm.result?.toUpperCase()}</span>
                                        </div>
                                        {selectedHistory.qc_reviewer && <div className="text-slate-500">复核人：{selectedHistory.qc_reviewer}</div>}
                                        {selectedHistory.qc_timestamp && <div className="text-slate-400">{selectedHistory.qc_timestamp}</div>}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* 控制操作卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300 flex-1 flex flex-col">
                        <div className="space-y-2 mt-auto">
                            {viewMode === 'history' ? (
                                <button onClick={() => { setViewMode('live'); setSelectedHistory(null); }} className="w-full h-12 bg-amber-500 text-white rounded font-bold text-[15px] shadow flex items-center justify-center hover:bg-amber-600 transition-colors">
                                    返回实时监控画面
                                </button>
                            ) : (
                                <div className="p-4 bg-slate-50 rounded border border-slate-200 text-center text-[14px] text-slate-400 font-bold italic">
                                    实时检测进行中...
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
            
            <style dangerouslySetInnerHTML={{ __html: pageStyles }} />
        </div>
    );
}
