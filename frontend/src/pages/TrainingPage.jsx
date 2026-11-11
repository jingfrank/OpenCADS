import React, { useState, useEffect, useRef } from 'react';
import {
    Plus, Upload, X, Trash2, Image as ImageIcon,
    Loader2, Calendar, HardDrive, Images, FolderOpen,
    ChevronRight, RefreshCw, Layers, Download, CheckCircle2,
    AlertCircle,
    Play, Square, Save, Info, Search
} from 'lucide-react';

export default function TrainingPage() {
    // 基础状态
    const [loading, setLoading] = useState(false);
    const [datasets, setDatasets] = useState([]);
    const [models, setModels] = useState([]);
    const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, running, finished, error
    const [trainingStage, setTrainingStage] = useState('');
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState([]);
    const logContainerRef = useRef(null);

    // FastAD 参数设置
    const [config, setConfig] = useState({
        dataset: '',
        onlyNormal: true,
        modelType: 'FastAD',
        backbone: 'wide_resnet50_2',
        coresetRatio: 0.1,
        kNeighbors: 9,
        customParams: ''
    });

    // 获取真实数据集列表和模型列表
    useEffect(() => {
        const fetchDatasets = async () => {
            try {
                const res = await fetch('/api/datasets');
                const data = await res.json();
                const fetchedDatasets = data.datasets || [];
                setDatasets(fetchedDatasets);
                if (fetchedDatasets.length > 0) {
                    setConfig(prev => ({ ...prev, dataset: fetchedDatasets[0].name }));
                }
            } catch (err) {
                console.error('Failed to fetch datasets:', err);
            }
        };
        const fetchModels = async () => {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                setModels(data.models_detail || []);
            } catch (err) {
                console.error('Failed to fetch models:', err);
            }
        };
        fetchDatasets();
        fetchModels();
    }, []);

    // 获取当前选中的数据集对象
    const selectedDataset = datasets.find(d => d.name === config.dataset);

    // 轮询训练状态
    useEffect(() => {
        const fetchTrainingStatus = async () => {
            try {
                const res = await fetch('/api/train/status');
                const data = await res.json();
                const prevStatus = trainingStatus;
                setTrainingStatus(data.state);
                setProgress(data.progress || 0);
                setTrainingStage(data.stage || '');
                // Refresh model list when training just finished
                if (prevStatus === 'running' && data.state === 'finished') {
                    const mRes = await fetch('/api/models');
                    const mData = await mRes.json();
                    setModels(mData.models_detail || []);
                }
            } catch (err) {
                console.error('Failed to fetch training status:', err);
            }
        };

        // 初始加载
        fetchTrainingStatus();

        // 训练中时高频轮询，否则低频
        const intervalId = setInterval(fetchTrainingStatus, trainingStatus === 'running' ? 1500 : 5000);
        return () => clearInterval(intervalId);
    }, [trainingStatus]);

    // 自动轮询系统日志
    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const res = await fetch('/api/system/logs?lines=50');
                const data = await res.json();
                if (data.logs) {
                    setLogs(data.logs);
                }
            } catch (err) {
                console.error('Failed to fetch system logs:', err);
            }
        };

        // 初始加载
        fetchLogs();

        // 开启轮询
        const intervalId = setInterval(fetchLogs, 2000);
        return () => clearInterval(intervalId);
    }, []);

    // 自动滚动日志
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const handleStartTraining = async () => {
        if (!config.dataset) {
            alert("请先选择一个数据集");
            return;
        }

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dataset_name: config.dataset,
                    backbone: config.backbone,
                    coreset_ratio: config.coresetRatio,
                    n_neighbors: config.kNeighbors
                }),
            });

            if (response.ok) {
                setTrainingStatus('running');
                setProgress(0);
            } else {
                const errData = await response.json();
                alert('启动训练失败: ' + (errData.detail || '未知错误'));
            }
        } catch (error) {
            alert('启动训练失败');
        }
    };

    const handleStopTraining = () => {
        // Note: backend doesn't support cancellation yet, this only resets frontend state
        setTrainingStatus('idle');
        setProgress(0);
        setTrainingStage('');
    };

    return (
        <div className="flex flex-col h-full overflow-hidden bg-slate-100/30">
            {/* 顶层面包屑式导航 */}
            <nav className="px-4 py-2 text-[20px] flex items-center bg-transparent border-b border-slate-300">
                <span className="text-blue-600 font-semibold cursor-pointer hover:underline">模型训练</span>
                <ChevronRight size={13} className="mx-1.5 text-slate-400" />
                <span className="text-black font-bold text-[20px]">FastAD 模型训练</span>
            </nav>

            {/* 三栏布局区域 */}
            <div className="flex-1 p-3 pt-1 flex space-x-2 overflow-hidden">
                
                {/* 第一栏：参数设置 (左) */}
                <div className="w-72 flex-shrink-0 flex flex-col space-y-2 overflow-y-auto pr-1 custom-scrollbar">
                    {/* 数据集选择卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            数据集选择
                        </h3>
                        <div className="space-y-3">
                            <div>
                                <label className="block text-[14px] font-bold text-black mb-1">选择训练数据集</label>
                                <select 
                                    className="w-full h-8 text-[14px] border border-slate-300 rounded bg-white text-black px-2 outline-none focus:border-blue-500"
                                    value={config.dataset}
                                    onChange={(e) => setConfig({...config, dataset: e.target.value})}
                                >
                                    {datasets.length === 0 ? (
                                        <option value="">暂无可用数据集</option>
                                    ) : (
                                        datasets.map(d => (
                                            <option key={d.name} value={d.name}>{d.name}</option>
                                        ))
                                    )}
                                </select>
                            </div>
                            <div className="flex items-center">
                                <input 
                                    type="checkbox" 
                                    id="onlyNormal" 
                                    className="w-3.5 h-3.5 mr-2 cursor-pointer"
                                    checked={config.onlyNormal}
                                    onChange={(e) => setConfig({...config, onlyNormal: e.target.checked})}
                                />
                                <label htmlFor="onlyNormal" className="text-[14px] text-black font-medium cursor-pointer">仅使用正常样本训练</label>
                            </div>
                            <div className="bg-slate-50 p-2 rounded border border-slate-200 text-[12px] text-black leading-relaxed">
                                {selectedDataset ? (
                                    <>
                                        原始样本: {selectedDataset.image_count},  <br/>
                                        数据大小: {selectedDataset.size}
                                    </>
                                ) : (
                                    "请先在图像数据管理中创建数据集"
                                )}
                            </div>
                        </div>
                    </div>

                    {/* 算法配置卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300 flex-1">
                        <h3 className="text-[18px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            FastAD 算法参数配置
                        </h3>
                        <div className="space-y-3">
                            <div>
                                <label className="block text-[14px] font-bold text-black mb-1">模型类型:</label>
                                <select className="w-full h-8 text-[14px] border border-slate-300 rounded bg-white text-black px-2 outline-none">
                                    <option>FastAD</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-[14px] font-bold text-black mb-1">主干网络:</label>
                                <select 
                                    className="w-full h-8 text-[14px] border border-slate-300 rounded bg-white text-black px-2 outline-none"
                                    value={config.backbone}
                                    onChange={(e) => setConfig({...config, backbone: e.target.value})}
                                >
                                    <option value="wide_resnet50_2">WideResNet-50</option>
                                    <option value="resnet18">ResNet-18</option>
                                </select>
                            </div>
                            <div className="flex items-center justify-between">
                                <label className="text-[14px] font-bold text-black">离散化参数:</label>
                                <input 
                                    type="number" 
                                    step="0.01"
                                    min="0.01"
                                    max="1"
                                    className="w-16 h-7 border border-slate-300 rounded px-2 text-[14px] text-center font-bold" 
                                    value={config.coresetRatio}
                                    onChange={(e) => setConfig({...config, coresetRatio: parseFloat(e.target.value)})}
                                />
                            </div>
                            <div className="grid grid-cols-2 gap-2 pt-2">
                                <button className="btn btn-outline h-8 text-[13px] font-bold flex items-center justify-center">
                                    <RefreshCw className="w-3 h-3 mr-1.5" />参数重置
                                </button>
                                <button className="btn btn-steel h-8 text-[13px] font-bold flex items-center justify-center shadow-sm">
                                    <Save className="w-3 h-3 mr-1.5" />参数保存
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* 第二栏：可视化监控 (中) */}
                <div className="flex-1 flex flex-col space-y-2 min-w-0 overflow-hidden">
                    {/* 构建流程卡片 */}
                    <div className="column-card flex-1 p-4 shadow-sm border border-slate-300 flex flex-col min-h-0 overflow-hidden">
                        <h3 className="text-[18px] font-extrabold text-black mb-4 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            模型训练状态
                        </h3>
                        
                        <div className="flex-1 flex flex-col overflow-y-auto custom-scrollbar">
                            <div className="mb-6">
                                {/* 状态步骤 */}
                                <div className="space-y-6">
                                <h4 className="text-[15px] font-extrabold text-black mb-4">
                                    模型训练步骤
                                        {trainingStatus === 'running' && <span className="ml-2 text-blue-600 text-[13px] animate-pulse">运行中...</span>}
                                        {trainingStatus === 'finished' && <span className="ml-2 text-green-600 text-[13px]">✓ 已完成</span>}
                                        {trainingStatus === 'error' && <span className="ml-2 text-red-600 text-[13px]">✗ 失败</span>}
                                    </h4>
                                    <div className="relative pl-6 space-y-8 border-l-2 border-slate-200 ml-2">
                                        {/* Step 1: 数据加载 */}
                                        {(() => {
                                            const steps = [
                                                { name: '图像预处理与数据加载', threshold: 15 },
                                                { name: 'FastAD特征提取', threshold: 60 },
                                                { name: '局部与全局统计建模', threshold: 85 },
                                                { name: '保存模型', threshold: 100 },
                                            ];
                                            return steps.map((step, idx) => {
                                                const isCompleted = progress > step.threshold || (progress === 100 && idx === steps.length - 1);
                                                const isActive = !isCompleted && progress >= (idx === 0 ? 0 : steps[idx - 1].threshold);
                                                const isWaiting = !isCompleted && !isActive;
                                                return (
                                                    <div key={idx} className={`relative min-h-[24px] ${isWaiting ? 'opacity-60' : ''}`}>
                                                        <div className={`absolute -left-[35px] top-0 w-6 h-6 rounded-full flex items-center justify-center ${
                                                            isCompleted ? 'bg-blue-600' : isActive ? 'bg-blue-600 border-4 border-white shadow-sm animate-pulse' : 'bg-white border-2 border-slate-300'
                                                        }`}>
                                                            {isCompleted && <CheckCircle2 className="w-4 h-4 text-white" />}
                                                        </div>
                                                        <div className="pl-1 flex flex-col justify-start">
                                                            {isActive ? (
                                                                <>
                                                                    <div className="flex justify-between items-center h-6 mb-1">
                                                                        <span className="text-[14px] font-bold text-blue-700">[PROGRESS] {step.name}</span>
                                                                        <span className="text-[14px] font-bold text-black">{progress}%</span>
                                                                    </div>
                                                                    <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                                                                        <div className="h-full bg-blue-600 transition-all duration-500" style={{width: `${progress}%`}}></div>
                                                                    </div>
                                                                    {trainingStage && <span className="text-[12px] text-slate-500 mt-1 font-bold">{trainingStage}</span>}
                                                                </>
                                                            ) : (
                                                                <div className="flex items-center h-6">
                                                                    <span className={`text-[14px] font-bold ${isCompleted ? 'text-black' : 'text-slate-400'}`}>
                                                                        [{isCompleted ? 'DONE' : 'WAIT'}] {step.name}
                                                                    </span>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            });
                                        })()}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 实时系统日志卡片 */}
                    <div className="column-card h-40 p-3 shadow-sm border border-slate-300 flex flex-col">
                        <div className="flex items-center justify-between border-b border-slate-200 pb-1 mb-2">
                            <h3 className="text-[15px] font-bold text-black flex items-center uppercase tracking-tight">
                                实时系统日志
                            </h3>
                            <div className="flex space-x-1">
                                <div className="w-2 h-2 rounded-full bg-red-400"></div>
                                <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
                                <div className="w-2 h-2 rounded-full bg-green-400"></div>
                            </div>
                        </div>
                        <div 
                            ref={logContainerRef}
                            className="flex-1 bg-slate-900 rounded p-2 font-mono text-[12px] text-green-400 overflow-y-auto leading-relaxed shadow-inner"
                        >
                            {logs.map((log, i) => (
                                <div key={i} className="mb-0.5"><span className="opacity-50">[{i+1}]</span> {log}</div>
                            ))}
                            <div className="animate-pulse">_</div>
                        </div>
                    </div>
                </div>

                {/* 第三栏：控制中心与版本 (右) */}
                <div className="w-64 flex-shrink-0 flex flex-col space-y-2">
                    {/* 已保存模型列表卡片 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300">
                        <h3 className="text-[15px] font-bold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            已保存模型
                        </h3>
                        <div className="space-y-2 max-h-60 overflow-y-auto pr-1 custom-scrollbar text-[14px]">
                            {models.length === 0 ? (
                                <div className="text-[13px] text-slate-400 text-center py-4">暂无已保存模型</div>
                            ) : (
                                models.map(model => (
                                    <div key={model.name} className="p-2 bg-white rounded border border-slate-200 hover:border-blue-300 transition-colors cursor-pointer">
                                        <div className="font-bold text-black text-[13px] truncate" title={model.name}>{model.name}</div>
                                        <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500">
                                            <span>{model.created_at ? new Date(model.created_at * 1000).toLocaleDateString() : '--'}</span>
                                            <span>{model.file_size ? `${(model.file_size / 1024 / 1024).toFixed(1)} MB` : '--'}</span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* 控制按钮区 */}
                    <div className="column-card p-3 shadow-sm border border-slate-300 flex-1">
                        <h3 className="text-[15px] font-bold text-black mb-3 flex items-center border-b border-slate-200 pb-2 uppercase tracking-tight">
                            模型构建与评估控制
                        </h3>
                        <div className="space-y-3">
                            <button 
                                onClick={handleStartTraining}
                                disabled={trainingStatus === 'running'}
                                className="w-full btn btn-green h-10 font-bold text-[14px] flex items-center justify-center uppercase tracking-widest shadow-md disabled:opacity-50"
                            >
                                <Play className="w-4 h-4 mr-2" /> 开始模型构建
                            </button>
                            <button 
                                onClick={handleStopTraining}
                                disabled={trainingStatus !== 'running'}
                                className="w-full btn btn-red h-10 font-bold text-[14px] flex items-center justify-center uppercase tracking-widest shadow-md disabled:opacity-50"
                            >
                                <Square className="w-4 h-4 mr-2 fill-current" /> 取消构建
                            </button>
                            <div className="bg-blue-600/10 p-3 rounded border border-blue-200 space-y-2">
                                <div className="flex items-center text-[12px] font-bold text-blue-700 uppercase">
                                    <span className="bg-blue-600 text-white px-1.5 py-0.5 rounded mr-2 text-[9px]">NEW</span>
                                    运行一次性验证评估
                                </div>
                                <div className="text-[13px] font-bold text-black leading-snug">
                                    部署状态: 运行一次性验证评估
                                </div>
                                <button className="w-full btn btn-steel h-8 text-[12px] font-bold flex items-center justify-center uppercase tracking-tighter">
                                    <Search className="w-3.5 h-3.5 mr-1.5" />保存当前版本模
                                </button>
                            </div>
                            <div className="pt-4 mt-auto">
                                <button className="w-full btn btn-steel h-10 font-bold text-[14px] flex items-center justify-center uppercase tracking-widest shadow-md">
                                    <Save className="w-4 h-4 mr-2" /> 保存当前版本
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* 自定义样式 */}
            <style jsx>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(0, 0, 0, 0.1);
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(0, 0, 0, 0.2);
                }
            `}</style>
        </div>
    );
}
