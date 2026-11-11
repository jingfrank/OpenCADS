import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function Layout() {
    return (
        <div className="h-screen bg-[#BFCFDF] flex flex-col overflow-hidden">
            <Sidebar />
            <div className="ml-[120px] flex-1 flex flex-col h-screen overflow-hidden">
                <Topbar />
                <main className="mt-[42px] flex-1 flex flex-col overflow-hidden relative">
                    <div className="absolute inset-0 flex flex-col">
                        <Outlet />
                    </div>
                </main>
            </div>
        </div>
    );
}
