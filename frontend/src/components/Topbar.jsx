import React from 'react';

export default function Topbar({ title = "芯片图像异常检测系统" }) {
    return (
        <header className="fixed top-0 left-[120px] right-0 bg-[#28446B] flex items-center px-4 text-white shadow-md h-[42px] z-40">
            <span className="font-bold tracking-tight opacity-90 text-[20px]">{title}</span>
        </header>
    );
}
