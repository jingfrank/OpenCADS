import React, { useState, useEffect, useRef } from 'react';
import { Upload, Loader2, Trash2, Play, FolderOpen, HardDrive, CheckCircle2 } from 'lucide-react';

export default function ModelsPage() {
    const [models, setModels] = useState([]);
    const [currentModel, setCurrentModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [loadingModel, setLoadingModel] = useState(null); // track which model is being loaded
    const uploadRef = useRef(null);

    const fetchModels = async () => {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            setModels(data.models_detail || []);
            setCurrentModel(data.current_model);
        } catch (err) {
            console.error('Failed to fetch models:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchModels();
        const interval = setInterval(fetchModels, 5000);
        return () => clearInterval(interval);
    }, []);

    const loadModel = async (name) => {
        setLoadingModel(name);
        try {
            const res = await fetch(`/api/models/load?model_name=${encodeURIComponent(name)}`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                fetchModels();
            } else {
                alert('加载模型失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            alert('加载模型失败');
        } finally {
            setLoadingModel(null);
        }
    };

    const deleteModel = async (name) => {
        if (!confirm(`确定要删除模型 "${name}" 吗？此操作不可恢复。`)) return;
        try {
            const res = await fetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                fetchModels();
            } else {
                alert('删除失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            alert('删除失败');
        }
    };

    const handleModelUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setUploading(true);
        try {
            const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success') {
                alert(data.message || '模型上传成功');
                fetchModels();
            } else {
                alert('上传失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            alert('上传失败');
        } finally {
            setUploading(false);
            if (uploadRef.current) uploadRef.current.value = '';
        }
    };

    // --- Utility formatters ---
    const formatFileSize = (bytes) => {
        if (!bytes) return '--';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    };

    const formatTime = (ts) => {
        if (!ts) return '--';
        const d = new Date(ts * 1000);
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    };

    const formatBackbone = (name) => {
        if (!name) return '未知';
        const map = {
            'wide_resnet50_2': 'WideResNet-50',
            'resnet18': 'ResNet-18',
            'resnet50': 'ResNet-50',
        };
        return map[name] || name;
    };

    const formatNumber = (val, digits = 2) => {
        if (val === null || val === undefined) return '--';
        return Number(val).toFixed(digits);
    };

    // Derive a version string from the model filename
    const deriveVersion = (name) => {
        // Try to extract version-like patterns, e.g. patchcore_v2.pth → v2
        const match = name.match(/v(\d+)/i);
        if (match) return `v${match[1]}`;
        // Otherwise derive from modification order
        return 'v1.0';
    };

    const isActive = (name) => currentModel === name;

    return (
        <div className="p-4">
            {/* Header */}
            <div className="mb-5 flex items-center justify-between">
                <div>
                    <h1 className="text-[20px] font-bold text-black flex items-center">
                        模型管理
                    </h1>
                    <p className="text-black mt-1 uppercase tracking-tight text-[11px]">Model Repository & Weights Management</p>
                </div>
                <div className="flex items-center gap-3">
                    {/* Import button */}
                    <input
                        ref={uploadRef}
                        type="file"
                        accept=".pth"
                        onChange={handleModelUpload}
                        disabled={uploading}
                        className="hidden"
                        id="model-upload-input"
                    />
                    <button
                        onClick={() => uploadRef.current?.click()}
                        disabled={uploading}
                        className="btn btn-outline px-4 py-2 text-[12px] font-bold uppercase tracking-tight"
                    >
                        {uploading ? (
                            <><Loader2 className="w-4 h-4 mr-1.5 animate-spin" />上传中...</>
                        ) : (
                            <><Upload className="w-4 h-4 mr-1.5" />导入离线权重</>
                        )}
                    </button>
                </div>
            </div>

            {/* Stats bar */}
            <div className="flex items-center gap-6 mb-5 px-1 text-[14px] text-black">
                <span className="flex items-center gap-1.5">
                    <HardDrive className="w-3.5 h-3.5 opacity-60" />
                    共 <b>{models.length}</b> 个模型
                </span>
                {currentModel && (
                    <span className="flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
                        当前激活: <b className="text-blue-700">{currentModel}</b>
                    </span>
                )}
            </div>

            {/* Model cards grid */}
            {loading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
                </div>
            ) : models.length === 0 ? (
                <div className="text-center py-20 column-card">
                    <FolderOpen className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                    <p className="text-slate-500 font-bold text-[13px] uppercase tracking-widest">暂无可用模型</p>
                    <p className="text-[12px] text-slate-400 mt-1">请先通过训练模块生成，或点击右上角导入离线权重</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {models.map(model => {
                        const active = isActive(model.name);
                        const isLoading = loadingModel === model.name;
                        return (
                            <div
                                key={model.name}
                                className={`column-card flex flex-col transition-all duration-200 group relative
                                    ${active
                                        ? 'ring-2 ring-blue-400 border-blue-300 shadow-md'
                                        : 'hover:shadow-md hover:border-slate-400'}`}
                            >
                                {/* Card header */}
                                <div className={`px-4 py-3 border-b flex items-center justify-between
                                    ${active ? 'bg-blue-50/80 border-blue-200' : 'bg-slate-50/50 border-slate-200'}`}>
                                    <div className="flex items-center gap-2.5 min-w-0">
                                        {active ? (
                                            <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse flex-shrink-0"></div>
                                        ) : (
                                            <div className="w-2.5 h-2.5 rounded-full bg-slate-300 flex-shrink-0"></div>
                                        )}
                                        <div className="min-w-0">
                                            <h3 className={`text-[15px] font-bold tracking-tight truncate ${active ? 'text-blue-800' : 'text-black'}`}>
                                                {model.name.replace('.pth', '')}
                                            </h3>
                                            {active && (
                                                <span className="text-[11px] font-extrabold text-blue-500 uppercase tracking-widest">
                                                    当前推理模型
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <span className="text-[12px] font-bold text-slate-400 bg-white/60 px-2 py-0.5 rounded border border-slate-200 flex-shrink-0">
                                        {deriveVersion(model.name)}
                                    </span>
                                </div>

                                {/* Card body - metadata grid */}
                                <div className="px-4 py-3 flex-1">
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-[14px]">
                                        {/* Creation time */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">创建时间</div>
                                                <div className="text-black font-semibold">{formatTime(model.created_at)}</div>
                                            </div>
                                        </div>
                                        {/* File size */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">文件大小</div>
                                                <div className="text-black font-semibold">{formatFileSize(model.file_size)}</div>
                                            </div>
                                        </div>
                                        {/* Backbone */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">模型结构</div>
                                                <div className="text-black font-semibold">{formatBackbone(model.backbone)}</div>
                                            </div>
                                        </div>
                                        {/* Coreset ratio */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">离散化参数</div>
                                                <div className="text-black font-semibold">
                                                    {model.coreset_ratio != null ? formatNumber(model.coreset_ratio, 2) : '--'}
                                                </div>
                                            </div>
                                        </div>
                                        {/* Threshold */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">异常阈值</div>
                                                <div className="text-black font-semibold">{formatNumber(model.threshold, 2)}</div>
                                            </div>
                                        </div>
                                        {/* Memory bank size */}
                                        <div className="flex items-start">
                                            <div>
                                                <div className="text-slate-400 text-[12px] uppercase tracking-wider font-medium">
                                                    {model.memory_bank_size != null ? '特征库' : '性能指标'}
                                                </div>
                                                <div className="text-black font-semibold">
                                                    {model.memory_bank_size != null
                                                        ? `${model.memory_bank_size.toLocaleString()} 条`
                                                        : <span className="text-slate-300 italic font-normal">待评估</span>}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Card footer - actions */}
                                <div className={`px-4 py-2.5 border-t flex items-center justify-end gap-2 ${active ? 'bg-blue-50/40 border-blue-100' : 'bg-slate-50/30 border-slate-200'}`}>
                                    {active && (
                                        <span className="text-[13px] text-blue-500 font-bold uppercase tracking-wider mr-auto flex items-center gap-1">
                                            <CheckCircle2 className="w-3.5 h-3.5" /> 已激活
                                        </span>
                                    )}
                                    <button
                                        onClick={() => deleteModel(model.name)}
                                        className="btn btn-outline px-3 py-1.5 text-[13px] font-bold text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                                        title="删除模型"
                                    >
                                        <Trash2 className="w-3 h-3 mr-1 opacity-70" />删除
                                    </button>
                                    {!active && (
                                        <button
                                            onClick={() => loadModel(model.name)}
                                            disabled={isLoading}
                                            className="btn btn-outline px-3 py-1.5 text-[13px] font-bold uppercase tracking-tight"
                                        >
                                            {isLoading ? (
                                                <><Loader2 className="w-3 h-3 mr-1 animate-spin" />加载中</>
                                            ) : (
                                                <><Play className="w-3 h-3 mr-1 opacity-70" />加载模型</>
                                            )}
                                        </button>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
