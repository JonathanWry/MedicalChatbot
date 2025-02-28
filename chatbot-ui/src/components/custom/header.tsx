import { ThemeToggle } from "./theme-toggle";
import React, {useState} from "react";
import {Sidebar} from "@/components/custom/sidebar.tsx";
import {MenuIcon} from "@/components/custom/icons.tsx";
import {Button} from "@/components/ui/button.tsx";


export const Header = () => {
    const [isSidebarOpen, setSidebarOpen] = useState(true);
    const [model, setModel] = useState("GPT-4");
  return (
    <>
      <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full">
        <div className="flex items-center space-x-1 sm:space-x-2">
            <Button variant="outline" size="icon" onClick={() => setSidebarOpen(true)}>
                <MenuIcon className="h-5 w-5" />
            </Button>
            <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="border rounded-md px-2 py-1 bg-white dark:bg-gray-800 dark:text-white"
            >
                <option value="GPT-4">GPT-4</option>
                <option value="GPT-3.5">GPT-3.5</option>
                <option value="Custom">Custom</option>
            </select>
            <ThemeToggle />
        </div>
      </header>
        <Sidebar isOpen={isSidebarOpen} onClose={() => setSidebarOpen(false)} />
    </>
  );
};