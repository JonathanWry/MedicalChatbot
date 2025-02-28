import React, {useEffect, useState} from "react";
import { Button } from "@/components/ui/button";
import { PlusCircle, MessageCircle, X, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import {ThemeToggle} from "@/components/custom/theme-toggle.tsx";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onDeleteChat?: (chatId: string) => void;
}


export function Sidebar({ isOpen, onClose, onDeleteChat }: SidebarProps) {
  const [chats, setChats] = useState<{ id: string; name: string; active: boolean }[]>([]);
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [topP, setTopP] = useState(0.3);
  const [temperature, setTemperature] = useState(0.4);
  const [topK, setTopK] = useState(3);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4o-mini");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const storedApiKey = localStorage.getItem("openai_api_key");
    if (storedApiKey) {
      setApiKey(storedApiKey);
    }
  }, []);
  function handleModelChange(newModel: string) {
    setModel(newModel);
    localStorage.setItem("chat_model", newModel);
  }
  function handleTopKChange(newTopK: number) {
    setTopK(newTopK);
    localStorage.setItem("chat_topK", String(newTopK));
  }

  function handleTemperatureChange(newTemp: number) {
    setTemperature(newTemp);
    localStorage.setItem("chat_temperature", String(newTemp));
  }

  function handleTopPChange(newTopP: number) {
    setTopP(newTopP);
    localStorage.setItem("chat_topP", String(newTopP));
  }

  const createNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      name: `Chat ${chats.length + 1}`,
      active: false,
    };
    setChats([...chats, newChat]);
  };

  const selectChat = (chatId: string) => {
    setActiveChat(chatId);
    setChats(chats.map(chat => ({
      ...chat,
      active: chat.id === chatId,
    })));
  };

  const handleDeleteChat = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();

    // Remove from local state first
    setChats((prevChats) => prevChats.filter(chat => chat.id !== chatId));

    // If onDeleteChat is provided, call it as well
    if (onDeleteChat) {
      onDeleteChat(chatId);
    }

    // Reset active chat if it was deleted
    if (activeChat === chatId) {
      setActiveChat(null);
    }
  };
  const handleSaveApiKey = async () => {
    setErrorMessage(""); // Clear previous errors
    try {
      const response = await fetch("http://localhost:8000/set_api_key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to set API key");
      }

      console.log("API key updated successfully");

      // Save API key in localStorage for persistence
      localStorage.setItem("openai_api_key", apiKey);
      console.log(apiKey);

      // Close modal
      setIsModalOpen(false);
    } catch (error: any) {
      console.error("Error setting API key:", error);
      setErrorMessage(error.message);
    }
  };

  return (
      <>
        {/* Sidebar Panel */}
        <div
            className={cn(
                "fixed inset-y-0 left-0 w-64 bg-background border-r transform transition-transform duration-200 ease-in-out z-50",
                isOpen ? "translate-x-0" : "-translate-x-full"
            )}
        >
          {/* Model Selection, Theme Toggle, and Close Button in One Row */}
          <div className="p-4 border-b border-gray-300 flex flex-row items-center gap-2 justify-between w-full">
            {/* Model Selection Dropdown */}
            <select
                value={model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="max-w-[70%] border rounded-md px-2 py-1 bg-white dark:bg-gray-800 dark:text-white"
            >
              <option value="gpt-4o-mini">gpt-4o-mini</option>
              <option value="GPT-3.5">GPT-3.5</option>
              <option value="Custom">Custom</option>
            </select>

            {/* Theme Toggle Button */}
            <div className="flex-shrink-0">
              <ThemeToggle />
            </div>

            {/* Close Sidebar Button (Right-aligned) */}
            <Button variant="ghost" size="icon" onClick={onClose} className="ml-auto">
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex flex-col h-full p-4">

            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Chats</h2>
            </div>

            <Button
                onClick={createNewChat}
                className="mb-4 flex items-center gap-2"
                variant="outline"
            >
              <PlusCircle className="h-4 w-4" />
              New Chat
            </Button>

            <ScrollArea className="flex-1 max-h-[35vh]">
              <div className="space-y-2">
                {chats.map((chat) => (
                    <div key={chat.id} className="group relative">
                      <Button
                          variant={chat.active ? "secondary" : "ghost"}
                          className="w-full justify-start gap-2 pr-8"
                          onClick={() => selectChat(chat.id)}
                      >
                        <MessageCircle className="h-4 w-4" />
                        {chat.name}
                      </Button>
                      <Button
                          variant="ghost"
                          size="icon"
                          className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={(e) => handleDeleteChat(e, chat.id)}
                      >
                        <Trash2 className="h-4 w-4 text-primary" />
                      </Button>
                    </div>
                ))}
              </div>
            </ScrollArea>

            {/* Fixed Settings Section */}
            <div className="p-4 border-t border-gray-500">
              <h2 className="text-sm text-gray-400 uppercase tracking-wide mb-3">Settings</h2>

              {/* Top-P Slider */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-1">Top-P: {topP.toFixed(1)}</label>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={topP}
                    onChange={(e) => handleTopPChange(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                />
              </div>

              {/* Temperature Slider */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-1">Temperature: {temperature.toFixed(1)}</label>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                />
              </div>

              {/* Top-K Dropdown */}
              <div className="mb-4">
                <label className="block text-sm text-gray-300 mb-1">Top-K</label>
                <select
                    value={topK}
                    onChange={(e) => handleTopKChange(parseInt(e.target.value))}
                    className="w-full p-2 rounded-md border border-gray-500"
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={15}>15</option>
                </select>
              </div>

              {/* Set API Key Button */}
              <div className="mb-4">
                <Button className="w-full bg-green-600 hover:bg-green-500 text-white" onClick={() => setIsModalOpen(true)}>
                  Set OpenAI Key
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* API Key Modal (Centered in Screen) */}
        {isModalOpen && (
            <div className="fixed inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 dark:bg-gray-950 transition-opacity z-50">
              <div className="bg-white p-6 rounded-lg shadow-lg w-96">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Enter OpenAI Key</h3>
                <input
                    type="text"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Paste API Key here..."
                    className="w-full p-2 border border-gray-300 rounded-md mb-4"
                />
                {/* Show error message if API key is invalid */}
                {errorMessage && (
                    <p className="text-red-500 text-sm mb-2">{errorMessage}</p>
                )}
                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setIsModalOpen(false)}>
                    Cancel
                  </Button>
                  <Button className="bg-green-600 hover:bg-green-500 text-white" onClick={handleSaveApiKey}>
                    Save
                  </Button>
                </div>
              </div>
            </div>
        )}
      </>
  );
}
