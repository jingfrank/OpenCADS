import React, { useState, useEffect, useCallback } from 'react';
import {
    Plus, Upload, X, Trash2, Image as ImageIcon,
    Loader2, Calendar, HardDrive, Images, FolderOpen,
    ChevronRight, RefreshCw, Download, CheckCircle2, AlertCircle
} from 'lucide-react';

export default function DatasetsPage() {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [categoryName, setCategoryName] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadLabel, setUploadLabel] = useState('none'); // none, normal, anomaly

    // 两层数据管理：选中的数据集
    const [selectedDataset, setSelectedDataset] = useState(null);
    const [datasetImages, setDatasetImages] = useState([]);
    const [loadingImages, setLoadingImages] = useState(false);
    const [currentImage, setCurrentImage] = useState(null);
    const [filterMode, setFilterMode] = useState('all'); // all, normal, anomaly
    const [selectedImages, setSelectedImages] = useState(new Set());
    const [imageResolution, setImageResolution] = useState(null);

    useEffect(() => {
        setImageResolution(null);
    }, [currentImage]);

    const fetchDatasets = async () => {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            setDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to fetch datasets:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDatasets();
    }, []);

    const fetchDatasetImages = async (datasetName) => {
        setLoadingImages(true);
        try {
            const res = await fetch(`/api/datasets/${encodeURIComponent(datasetName)}/images`);
            const data = await res.json();
            setDatasetImages(data.images || []);
            if (data.images && data.images.length > 0) {
                setCurrentImage(data.images[0]);
            } else {
                setCurrentImage(null);
            }
        } catch (err) {
            console.error('Failed to fetch images:', err);
        } finally {
            setLoadingImages(false);
        }
    };

    const handleSelectDataset = (dataset) => {
        setSelectedDataset(dataset);
        fetchDatasetImages(dataset.name);
    };

    const handleBackToList = () => {
        setSelectedDataset(null);
        setDatasetImages([]);
        setCurrentImage(null);
    };

    const handleLabelChange = async (value) => {
        if (!currentImage || !selectedDataset) return;
        
        let queryParams = '';
        if (value === 'anomaly') queryParams = '?is_anomaly=true';
        else if (value === 'normal') queryParams = '?is_anomaly=false';

        try {
            const res = await fetch(`/api/datasets/${encodeURIComponent(selectedDataset.name)}/images/${encodeURIComponent(currentImage.filename)}/label${queryParams}`, {
                method: 'POST'
            });
            const data = await res.json();
            if (data.status === 'success') {
                const newValue = value === 'anomaly' ? true : value === 'normal' ? false : null;
                setCurrentImage(prev => ({ ...prev, is_anomaly: newValue }));
                setDatasetImages(prev => prev.map(img =>
                    img.filename === currentImage.filename ? { ...img, is_anomaly: newValue } : img
                ));
            }
        } catch (err) {
            console.error('Failed to update label', err);
        }
    };

    const handleBatchUpload = () => {
        setCategoryName(selectedDataset.name);
        setShowModal(true);
    };

    const handleSelectAll = () => {
        // Select all currently visible (filtered) images
        const filtered = filterMode === 'normal'
            ? datasetImages.filter(img => img.is_anomaly === false)
            : filterMode === 'anomaly'
                ? datasetImages.filter(img => img.is_anomaly === true)
                : filterMode === 'unlabeled'
                    ? datasetImages.filter(img => img.is_anomaly === null || img.is_anomaly === undefined)
                    : datasetImages;
        setSelectedImages(new Set(filtered.map(img => img.filename)));
    };

    const handleDeselectAll = () => {
        setSelectedImages(new Set());
    };

    const toggleImageSelect = (filename) => {
        setSelectedImages(prev => {
            const next = new Set(prev);
            if (next.has(filename)) {
                next.delete(filename);
            } else {
                next.add(filename);
            }
            return next;
        });
    };

    const handleBatchLabel = async (isAnomaly) => {
        if (selectedImages.size === 0) {
            alert('请先选择图片');
            return;
        }
        const label = isAnomaly === true ? '异常' : isAnomaly === false ? '正常' : '未标记';
        if (!confirm(`确定将选中的 ${selectedImages.size} 张图片标记为「${label}」吗？`)) return;

        try {
            const promises = [...selectedImages].map(filename => {
                let queryParams = '';
                if (isAnomaly === true) queryParams = '?is_anomaly=true';
                else if (isAnomaly === false) queryParams = '?is_anomaly=false';
                
                return fetch(`/api/datasets/${encodeURIComponent(selectedDataset.name)}/images/${encodeURIComponent(filename)}/label${queryParams}`, {
                    method: 'POST'
                });
            });
            await Promise.all(promises);
            // Update local state
            setDatasetImages(prev => prev.map(img =>
                selectedImages.has(img.filename) ? { ...img, is_anomaly: isAnomaly } : img
            ));
            if (currentImage && selectedImages.has(currentImage.filename)) {
                setCurrentImage(prev => ({ ...prev, is_anomaly: isAnomaly }));
            }
            setSelectedImages(new Set());
        } catch (err) {
            alert('批量标记失败');
        }
    };

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSelectedFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleFileSelect = (e) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!categoryName.trim()) {
            alert('请输入类别名称');
            return;
        }
        if (!selectedFile) {
            alert('请选择文件');
            return;
        }

        setUploading(true);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', selectedFile);

        // Simulate progress
        const progressInterval = setInterval(() => {
            setUploadProgress(prev => {
                if (prev >= 90) {
                    clearInterval(progressInterval);
                    return prev;
                }
                return prev + 10;
            });
        }, 200);

        try {
            let uploadUrl = `/api/datasets?category=${encodeURIComponent(categoryName)}`;
            if (uploadLabel !== 'none') {
                uploadUrl += `&label=${uploadLabel}`;
            }
            const res = await fetch(uploadUrl, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            clearInterval(progressInterval);
            setUploadProgress(100);

            if (data.status === 'success') {
                setTimeout(() => {
                    setShowModal(false);
                    setCategoryName('');
                    setSelectedFile(null);
                    setUploadProgress(0);
                    setUploadLabel('none');
                    fetchDatasets();
                    if (selectedDataset) {
                        fetchDatasetImages(selectedDataset.name);
                    }
                }, 500);
            } else {
                alert('上传失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            clearInterval(progressInterval);
            alert('上传失败');
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (e, name) => {
        e.stopPropagation(); // 阻止触发卡片点击
        if (!confirm(`确定要删除数据集 "${name}" 吗？此操作不可恢复。`)) {
            return;
        }

        try {
            const res = await fetch(`/api/datasets/${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            const data = await res.json();

            if (data.status === 'success') {
                setDatasets(prev => prev.filter(d => d.name !== name));
                if (selectedDataset && selectedDataset.name === name) {
                    setSelectedDataset(null);
                }
            } else {
                alert('删除失败: ' + (data.detail || '未知错误'));
            }
        } catch (err) {
            alert('删除失败');
        }
    };

    const closeModal = () => {
        if (!uploading) {
            setShowModal(false);
            setCategoryName(selectedDataset ? selectedDataset.name : '');
            setSelectedFile(null);
            setUploadProgress(0);
            setUploadLabel('none');
        }
    };

    const renderModal = () => {
        if (!showModal) return null;
        return (
            <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-[100] p-4">
                <div className="bg-white rounded shadow-2xl w-full max-w-md overflow-hidden border border-slate-300">
                    <div className="flex items-center justify-between px-6 py-4 bg-[#23364d] text-white">
                        <h2 className="text-sm font-bold tracking-widest flex items-center">
                            {selectedDataset ? '批量导入数据' : '新建数据集类别'}
                        </h2>
                        <button
                            onClick={closeModal}
                            disabled={uploading}
                            className="opacity-70 hover:opacity-100 transition-opacity disabled:opacity-30"
                        >
                            <X size={18} />
                        </button>
                    </div>

                    <div className="p-4 space-y-6">
                        <div>
                            <label className="block text-[11px] font-extrabold text-slate-500 mb-2 uppercase tracking-tighter">
                                类别标识 (ID NAME) <span className="text-red-500">*</span>
                            </label>
                            <input
                                type="text"
                                value={categoryName}
                                onChange={(e) => setCategoryName(e.target.value)}
                                placeholder="芯片A / 产品型号-01"
                                disabled={!!selectedDataset || uploading}
                                className="w-full h-10 px-4 border border-slate-300 bg-slate-50 focus:bg-white focus:ring-1 focus:ring-blue-500 outline-none transition-all text-sm font-medium disabled:opacity-70"
                            />
                        </div>

                        <div>
                            <label className="block text-[11px] font-extrabold text-slate-500 mb-2 uppercase tracking-tighter">
                                数据压缩包 (ATTACHMENT) <span className="text-red-500">*</span>
                            </label>
                            <div
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                                className={`relative border border-dashed rounded p-8 text-center transition-all ${dragActive
                                    ? 'border-blue-500 bg-blue-50'
                                    : 'border-slate-300 bg-slate-50 hover:bg-white'
                                    } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
                            >
                                <input
                                    type="file"
                                    accept=".zip,.png,.jpg,.jpeg,.bmp,.gif,.webp"
                                    onChange={handleFileSelect}
                                    disabled={uploading}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />

                                {selectedFile ? (
                                    <div className="flex flex-col items-center">
                                        <div className="w-12 h-12 bg-blue-100 rounded-sm flex items-center justify-center mb-3">
                                            <Upload className="w-6 h-6 text-blue-600" />
                                        </div>
                                        <p className="font-bold text-slate-700 text-sm truncate max-w-[250px]">
                                            {selectedFile.name}
                                        </p>
                                        <p className="text-[11px] text-slate-400 mt-1">
                                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                                        </p>
                                    </div>
                                ) : (
                                    <div>
                                        <Upload className="w-8 h-8 text-slate-300 mx-auto mb-3" />
                                        <p className="text-slate-500 font-bold text-xs">
                                            点击选择或拖拽文件到此处
                                        </p>
                                        <p className="text-[10px] text-slate-400 mt-2 italic">
                                            支持 ZIP / PNG / JPG 格式，最高 500MB
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {!uploading && (
                            <div>
                                <label className="block text-[11px] font-extrabold text-slate-500 mb-2 uppercase tracking-tighter">
                                    上传时默认标签设定 (DEFAULT LABEL)
                                </label>
                                <div className="flex gap-4">
                                    <label className="flex items-center gap-2 text-sm cursor-pointer hover:bg-slate-50 p-2 rounded border border-transparent hover:border-slate-200 transition-all">
                                        <input type="radio" name="uploadLabel" checked={uploadLabel === 'none'} onChange={() => setUploadLabel('none')} className="w-4 h-4 text-blue-600" />
                                        <span>不标记 (自动推断/未标记)</span>
                                    </label>
                                    <label className="flex items-center gap-2 text-sm cursor-pointer hover:bg-slate-50 p-2 rounded border border-transparent hover:border-slate-200 transition-all">
                                        <input type="radio" name="uploadLabel" checked={uploadLabel === 'normal'} onChange={() => setUploadLabel('normal')} className="w-4 h-4 text-green-600" />
                                        <span className="font-bold text-green-700">全体标记为正常</span>
                                    </label>
                                    <label className="flex items-center gap-2 text-sm cursor-pointer hover:bg-slate-50 p-2 rounded border border-transparent hover:border-slate-200 transition-all">
                                        <input type="radio" name="uploadLabel" checked={uploadLabel === 'anomaly'} onChange={() => setUploadLabel('anomaly')} className="w-4 h-4 text-red-600" />
                                        <span className="font-bold text-red-700">全体标记为异常</span>
                                    </label>
                                </div>
                            </div>
                        )}

                        {uploading && (
                            <div className="space-y-2">
                                <div className="flex items-center justify-between text-[11px]">
                                    <span className="text-slate-500 font-bold">UPLOADING...</span>
                                    <span className="text-blue-600 font-bold">{uploadProgress}%</span>
                                </div>
                                <div className="h-1 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                                    <div
                                        className="h-full bg-blue-600 transition-all duration-300"
                                        style={{ width: `${uploadProgress}%` }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="flex gap-4 p-4 bg-slate-50 border-t border-slate-200">
                        <button
                            onClick={closeModal}
                            disabled={uploading}
                            className="flex-1 py-2.5 border border-slate-300 text-slate-600 font-bold text-xs uppercase tracking-widest hover:bg-white transition-colors disabled:opacity-50"
                        >
                            取消
                        </button>
                        <button
                            onClick={handleUpload}
                            disabled={uploading || !categoryName.trim() || !selectedFile}
                            className="flex-1 py-2.5 bg-blue-600 text-white font-bold text-xs uppercase tracking-widest shadow-md hover:bg-blue-700 transition-all disabled:bg-slate-300 disabled:shadow-none disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {uploading ? (
                                <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    正在处理
                                </>
                            ) : (
                                <>
                                    <CheckCircle2 className="w-3.5 h-3.5" />
                                    确认导入
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <>
            {selectedDataset ? (
                // 第二层：详情页渲染 (三栏布局)
                <div className="flex flex-col h-full overflow-hidden">
                    {/* 导航路径 */}
                    <nav className="px-4 py-2 text-[20px] flex items-center bg-transparent border-b border-slate-300">
                        <span
                            className="text-blue-600 font-semibold cursor-pointer hover:underline"
                            onClick={handleBackToList}
                        >数据集管理</span>
                        <ChevronRight size={13} className="mx-1.5 text-slate-400" />
                        <span className="text-black font-bold text-[20px]">{selectedDataset.name}</span>
                    </nav>

                    {/* 三栏布局区域 */}
                    <div className="flex-1 p-3 pt-1 flex space-x-2 overflow-hidden">

                        {/* 第一栏：版本信息 (左) */}
                        <div className="w-64 flex-shrink-0 flex flex-col">
                            <div className="column-card p-4 flex flex-col">
                                <h3 className="text-[20px] font-extrabold border-b border-slate-300 pb-2 mb-4 text-black flex items-center tracking-tight">
                                    数据集信息与控制
                                </h3>
                                <div className="space-y-4 mt-2 mb-2">
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">数据集名称：</span>
                                        <span className="font-bold text-black text-[15px] truncate text-right ml-2" title={selectedDataset.name}>{selectedDataset.name}</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">创建时间：</span>
                                        <span className="text-black font-bold text-[15px]">{selectedDataset.created_at}</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">图片数量：</span>
                                        <span className="text-black font-bold text-[15px]">{selectedDataset.image_count}</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">总数据量：</span>
                                        <span className="text-black font-bold text-[15px]">{selectedDataset.size}</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">正常样本：</span>
                                        <span className="text-green-700 font-bold text-[15px]">{datasetImages.filter(img => !img.is_anomaly).length} 张</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-black font-medium text-[14px] whitespace-nowrap">异常样本：</span>
                                        <span className="text-red-600 font-bold text-[15px]">{datasetImages.filter(img => img.is_anomaly).length} 张</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* 第二栏：训练样本集 (中) */}
                        <div className="flex-1 column-card flex flex-col overflow-hidden">
                            <div className="p-3 border-b border-slate-300 bg-slate-50/50">
                                <h3 className="text-[20px] font-extrabold text-black mb-3 flex items-center border-b border-slate-200 pb-2 tracking-tight">
                                    训练样本集
                                </h3>

                                {/* 过滤选项 */}
                                <div className="flex flex-wrap items-center gap-3 mb-3">
                                    <div className="flex border border-slate-300 rounded overflow-hidden bg-white shadow-sm h-8">
                                        {[{key: 'all', label: '所有样本'}, {key: 'normal', label: '正常样本'}, {key: 'anomaly', label: '异常样本'}, {key: 'unlabeled', label: '未标记'}].map((f, idx) => (
                                            <button
                                                key={f.key}
                                                onClick={() => setFilterMode(f.key)}
                                                className={`px-4 text-[14px] font-medium h-full flex items-center justify-center transition-colors ${
                                                    filterMode === f.key ? 'bg-[#5c82a8] text-white' : 'hover:bg-slate-50 text-black'
                                                } ${idx > 0 ? 'border-l border-slate-300' : ''}`}
                                            >{f.label}</button>
                                        ))}
                                    </div>
                                    <span className="text-[13px] text-slate-500 font-medium">
                                        共 {datasetImages.length} 张，显示 {(() => {
                                            if (filterMode === 'normal') return datasetImages.filter(img => img.is_anomaly === false).length;
                                            if (filterMode === 'anomaly') return datasetImages.filter(img => img.is_anomaly === true).length;
                                            if (filterMode === 'unlabeled') return datasetImages.filter(img => img.is_anomaly === null || img.is_anomaly === undefined).length;
                                            return datasetImages.length;
                                        })()} 张
                                    </span>
                                </div>

                                {/* 操作按钮组 */}
                                <div className="flex justify-between items-center bg-slate-100/80 p-1.5 border border-slate-300 rounded-sm">
                                    <div className="flex space-x-1.5">
                                        <button onClick={handleBatchUpload} className="btn btn-steel px-3 py-1.5 shadow-sm text-[12px] font-bold"><Upload className="w-3.5 h-3.5 mr-1.5" />批量上传/拖拽</button>
                                    </div>
                                    <div className="flex space-x-1 items-center">
                                        {selectedImages.size > 0 && (
                                            <span className="text-[11px] text-blue-600 font-bold mr-1">已选 {selectedImages.size} 张</span>
                                        )}
                                        <button
                                            onClick={selectedImages.size > 0 ? handleDeselectAll : handleSelectAll}
                                            className="btn btn-outline px-2.5 py-1 text-[11px] bg-white text-slate-600 font-medium border-slate-300"
                                        >{selectedImages.size > 0 ? '取消全选' : '全选'}</button>
                                        <button
                                            onClick={() => handleBatchLabel(null)}
                                            disabled={selectedImages.size === 0}
                                            className="btn btn-outline px-3 py-1 text-[11px] font-bold border-slate-300 text-slate-600 disabled:opacity-40"
                                        ><AlertCircle className="w-3 h-3 mr-1" />设为未标记</button>
                                        <button
                                            onClick={() => handleBatchLabel(false)}
                                            disabled={selectedImages.size === 0}
                                            className="btn btn-green px-3 py-1 text-[11px] font-bold disabled:opacity-40"
                                        ><CheckCircle2 className="w-3 h-3 mr-1" />标记正常</button>
                                        <button
                                            onClick={() => handleBatchLabel(true)}
                                            disabled={selectedImages.size === 0}
                                            className="btn btn-red px-3 py-1 text-[11px] font-bold disabled:opacity-40"
                                        ><AlertCircle className="w-3 h-3 mr-1" />标记异常</button>
                                    </div>
                                </div>
                            </div>

                            {/* 滚动图片库区 */}
                            <div className="flex-1 overflow-y-auto p-3 bg-slate-400/5">
                                {loadingImages ? (
                                    <div className="flex items-center justify-center h-full">
                                        <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
                                    </div>
                                ) : datasetImages.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center h-full text-slate-400 space-y-2">
                                        <ImageIcon size={48} className="opacity-20" />
                                        <p className="text-sm">该数据集暂无图片</p>
                                    </div>
                                ) : (
                                    <div className="img-grid">
                                        {(() => {
                                            const filtered = filterMode === 'normal'
                                                ? datasetImages.filter(img => img.is_anomaly === false)
                                                : filterMode === 'anomaly'
                                                    ? datasetImages.filter(img => img.is_anomaly === true)
                                                    : filterMode === 'unlabeled'
                                                        ? datasetImages.filter(img => img.is_anomaly === null || img.is_anomaly === undefined)
                                                        : datasetImages;
                                            if (filtered.length === 0) {
                                                return (
                                                    <div className="col-span-full flex flex-col items-center justify-center py-12 text-slate-400 space-y-2">
                                                        <ImageIcon size={36} className="opacity-20" />
                                                        <p className="text-sm">没有符合筛选条件的图片</p>
                                                    </div>
                                                );
                                            }
                                            return filtered.map((img) => (
                                            <div
                                                key={img.id}
                                                className={`img-item shadow-sm ${currentImage && currentImage.id === img.id ? 'active' : ''}`}
                                                onClick={() => setCurrentImage(img)}
                                            >
                                                <input
                                                    type="checkbox"
                                                    className="absolute top-1 left-1 z-10 w-4 h-4 cursor-pointer"
                                                    checked={selectedImages.has(img.filename)}
                                                    onChange={(e) => {
                                                        e.stopPropagation();
                                                        toggleImageSelect(img.filename);
                                                    }}
                                                />
                                                <div className="aspect-square bg-slate-200 rounded-sm overflow-hidden relative border border-slate-300/50">
                                                    <img
                                                        src={`/api/datasets/${encodeURIComponent(selectedDataset.name)}/images/${img.filename}`}
                                                        className="w-full h-full object-cover"
                                                        alt={img.filename}
                                                    />
                                                    {img.is_anomaly && (
                                                        <div className="absolute inset-0 flex items-center justify-center bg-red-400/10 pointer-events-none">
                                                            <div className="w-full h-0.5 bg-red-600/70 rotate-[45deg]"></div>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="text-center mt-2 text-[12px] truncate px-1">
                                                    <span className={img.is_anomaly === true ? 'text-red-700 font-extrabold' : img.is_anomaly === false ? 'text-black font-bold' : 'text-slate-500 font-bold'}>
                                                        {img.filename} {img.is_anomaly === true ? '[异常]' : img.is_anomaly === false ? '[正常]' : '[未标记]'}
                                                    </span>
                                                </div>
                                            </div>
                                            ));
                                        })()}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* 第三栏：图像详情与编辑 (右) */}
                        <div className="w-80 flex-shrink-0 flex flex-col">
                            <div className="column-card flex-1 p-3 flex flex-col overflow-y-auto">
                                <h3 className="text-[20px] font-extrabold border-b border-slate-300 pb-2 mb-3 text-black flex items-center tracking-tight">
                                    图像详情与标注
                                </h3>

                                {/* 实时大图 */}
                                <div className="mb-3">
                                    {currentImage ? (
                                        <div className="w-full bg-[#1a2d42] rounded border border-slate-500 overflow-hidden shadow-inner flex items-center justify-center">
                                            <img
                                                src={`/api/datasets/${encodeURIComponent(selectedDataset.name)}/images/${currentImage.filename}`}
                                                className="w-full h-auto block"
                                                alt="Preview"
                                                onLoad={(e) => setImageResolution(`${e.target.naturalWidth}x${e.target.naturalHeight}`)}
                                            />
                                        </div>
                                    ) : (
                                        <div className="w-full aspect-[4/3] bg-[#1a2d42] rounded border border-slate-500 flex items-center justify-center text-slate-600 shadow-inner">
                                            <ImageIcon size={48} className="opacity-20" />
                                        </div>
                                    )}
                                </div>

                                {currentImage ? (
                                    <div className="space-y-4 flex-1 flex flex-col">
                                        <div className="space-y-2">
                                            <div className="flex items-baseline gap-1">
                                                <span className="text-black font-medium text-[14px] flex-shrink-0">图像名称：</span>
                                                <span className="font-bold text-black text-[14px] break-all">{currentImage.filename}</span>
                                            </div>
                                            <div className="flex items-baseline gap-1">
                                                <span className="text-black font-medium text-[14px] flex-shrink-0">分辨率：</span>
                                                <span className="font-bold text-black text-[14px]">{imageResolution || '加载中...'}</span>
                                            </div>
                                        </div>

                                        <div className="space-y-2 flex-1 flex flex-col">
                                            <div>
                                                <label className="text-[14px] font-medium text-black block mb-1">标签</label>
                                                <select
                                                    className={`w-full h-8 rounded-sm border-slate-400 font-bold text-[14px] shadow-sm ${currentImage.is_anomaly === true ? 'text-red-700 bg-red-50' : currentImage.is_anomaly === false ? 'text-green-700 bg-green-50' : 'text-slate-700 bg-slate-50'}`}
                                                    value={currentImage.is_anomaly === true ? 'anomaly' : currentImage.is_anomaly === false ? 'normal' : 'none'}
                                                    onChange={(e) => handleLabelChange(e.target.value)}
                                                >
                                                    <option value="none">未标记 (Unlabeled)</option>
                                                    <option value="normal">正常 (Normal)</option>
                                                    <option value="anomaly">异常 (Anomaly)</option>
                                                </select>
                                            </div>

                                            <div className="flex-1 flex flex-col min-h-0">
                                                <label className="text-[14px] font-medium text-black block mb-1">备注说明</label>
                                                <textarea
                                                    className="w-full flex-1 min-h-[80px] rounded-sm p-2 text-black font-medium border-slate-400 leading-relaxed text-[12px] focus:ring-1 focus:ring-slate-500 outline-none bg-white"
                                                    placeholder="在此记录检测详情或复核意见..."
                                                ></textarea>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex-1 flex items-center justify-center text-black font-bold text-sm italic">
                                        请选择图像进行查看
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                // 第一层：数据集网格渲染 (参考 dataset.html 风格)
                <div className="p-4">
                    {/* 页面标题和操作栏 */}
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-[20px] font-bold text-black flex items-center">
                                图像数据管理模块
                            </h1>
                            <p className="text-black mt-1 uppercase tracking-tight text-[11px]">Dataset Management & Version Control</p>
                        </div>
                        <button
                            onClick={() => setShowModal(true)}
                            className="flex items-center gap-2 px-6 py-2 bg-[#5c82a8] hover:bg-[#4a6b8a] text-white rounded shadow-md transition-all font-bold text-[12px]"
                        >
                            <Plus className="w-4 h-4" />
                            新建数据集
                        </button>
                    </div>

                    {/* 数据集卡片网格 */}
                    {loading ? (
                        <div className="flex items-center justify-center py-20">
                            <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
                        </div>
                    ) : datasets.length === 0 ? (
                        <div className="column-card p-12 text-center shadow-inner bg-slate-50">
                            <div className="w-16 h-16 bg-slate-200 rounded-full flex items-center justify-center mx-auto mb-4">
                                <FolderOpen className="w-8 h-8 text-slate-400" />
                            </div>
                            <h3 className="text-[13px] font-bold text-slate-700 mb-2">暂无可用数据集</h3>
                            <p className="text-slate-500 mb-6 text-[12px]">系统当前未检测到任何训练数据，请通过右上方按钮创建。</p>
                            <button
                                onClick={() => setShowModal(true)}
                                className="btn btn-steel px-6 py-2 shadow-sm"
                            >
                                <Plus className="w-4 h-4 mr-2" />
                                立即创建
                            </button>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                            {datasets.map((dataset) => (
                                <div
                                    key={dataset.name}
                                    className="column-card group cursor-pointer hover:shadow-xl transition-all duration-300 hover:-translate-y-1 relative"
                                    onClick={() => handleSelectDataset(dataset)}
                                >
                                    {/* 缩略图 */}
                                    <div className="aspect-video bg-slate-800 relative overflow-hidden">
                                        {dataset.thumbnail ? (
                                            <img
                                                src={`data:image/jpeg;base64,${dataset.thumbnail}`}
                                                alt={dataset.name}
                                                className="w-full h-full object-cover transition-transform group-hover:scale-105 duration-500 opacity-90 group-hover:opacity-100"
                                            />
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-600">
                                                <Images className="w-10 h-10" />
                                            </div>
                                        )}
                                        <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent"></div>

                                        {/* 删除按钮 */}
                                        <button
                                            onClick={(e) => handleDelete(e, dataset.name)}
                                            className="absolute top-2 right-2 p-1.5 bg-red-600/80 text-white rounded hover:bg-red-700 transition-opacity z-10 opacity-0 group-hover:opacity-100 shadow-lg"
                                            title="删除数据集"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>


                                    </div>

                                    {/* 信息 */}
                                    <div className="p-4 bg-white">
                                        <h3 className="font-bold text-black truncate mb-3 text-[15px] group-hover:text-blue-700 transition-colors">
                                            {dataset.name}
                                        </h3>
                                        <div className="space-y-2 text-[14px] text-black">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center">
                                                    <span>样本数量</span>
                                                </div>
                                                <span className="font-bold text-black">{dataset.image_count} 张</span>
                                            </div>
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center">
                                                    <span>最后更新</span>
                                                </div>
                                                <span className="font-medium text-black">{dataset.created_at}</span>
                                            </div>
                                        </div>
                                        <div className="mt-4 pt-3 border-t border-slate-100 flex justify-end">
                                            <span className="text-[13px] font-bold text-black group-hover:text-blue-500 flex items-center uppercase tracking-widest">
                                                进入管理 <ChevronRight className="w-3 h-3 ml-1" />
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                </div>
            )}
            {renderModal()}
        </>
    );
}
