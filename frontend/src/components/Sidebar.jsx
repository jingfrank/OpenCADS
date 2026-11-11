import React from 'react';
import { NavLink } from 'react-router-dom';
import fastadLogo from '../assets/fastad-logo.svg';

const navItems = [
    {
        name: '数据管理',
        path: '/datasets',
    },
    {
        name: '模型训练',
        path: '/training',
    },
    {
        name: '模型管理',
        path: '/models',
    },
    {
        name: '实时检测',
        path: '/realtime-detection',
    }
];

export default function Sidebar() {
    return (
        <aside className="fixed left-0 top-0 h-full w-[120px] bg-[#28446B] flex flex-col text-slate-300 z-50">
            <div className="h-[42px] flex items-center justify-center shadow-md">
                <img src={fastadLogo} alt="FastAD" className="w-[104px] h-[30px] object-contain" />
            </div>
            {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    title={item.name}
                    className={({ isActive }) =>
                        `flex items-center h-12 px-4 text-[15px] font-bold tracking-tight transition-all ${isActive
                            ? 'text-white border-l-2 border-white bg-white/10 pl-3.5'
                            : 'hover:text-white hover:bg-white/5'
                        }`
                    }
                >
                    {item.name}
                </NavLink>
            ))}
        </aside>
    );
}
