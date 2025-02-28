import {ChatInput} from "@/components/custom/chatinput";
import {PreviewMessage, ThinkingMessage} from "../../components/custom/message";
import {useScrollToBottom} from '@/components/custom/use-scroll-to-bottom';
import {useState, useRef, useEffect} from "react";
import {message} from "../../interfaces/interfaces"
import {Overview} from "@/components/custom/overview";
import {Header} from "@/components/custom/header";
import {v4 as uuidv4} from 'uuid';
import {Button} from "@/components/ui/button.tsx";
import { toast } from 'sonner';

const socket = new WebSocket("ws://localhost:8800"); //change to your websocket endpoint

export function Chat() {
    const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
    const [messages, setMessages] = useState<message[]>([]);
    const [question, setQuestion] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [apiKey, setApiKey] = useState<string | null>(null);

    const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(null);
    useEffect(() => {
        const storedApiKey = localStorage.getItem("openai_api_key");
        console.log(storedApiKey);
        if (storedApiKey) {
            setApiKey(storedApiKey);
        }
    }, []);

    // Provide a setter that also writes to localStorage if needed
    const handleSetApiKey = (newKey: string) => {
        setApiKey(newKey);
        localStorage.setItem("openai_api_key", newKey);
    };
    const cleanupMessageHandler = () => {
        if (messageHandlerRef.current && socket) {
            socket.removeEventListener("message", messageHandlerRef.current);
            messageHandlerRef.current = null;
        }
    };

    async function handleSubmit(text?: string, mode: string) {
        const model = localStorage.getItem("chat_model") || "gpt-4o-mini";
        console.log("model",model)
        const topK = parseInt(localStorage.getItem("chat_topK") || "3", 10);
        const temperature = parseFloat(localStorage.getItem("chat_temperature") || "0.4");
        const topP = parseFloat(localStorage.getItem("chat_topP") || "0.3");

        // console.log(mode);
        if (isLoading) return;

        const messageText = text || question;
        // console.log("Before setIsLoading: ", isLoading);
        setIsLoading(true);
        // console.log("After setIsLoading: ", isLoading);

        // Append user's message to chat
        const traceId = uuidv4();
        setMessages(prev => [...prev, {content: messageText, role: "user", id: traceId}]);
        setQuestion("");

        const body = {
            query: messageText,
            model: model,
            top_k: topK,
            temperature: temperature,
            top_p: topP,
            mode: mode,
        };

        try {
            const response = await fetch("http://localhost:8000/query", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(body),
            });

            if (!response.ok) throw new Error("Server error. Please try again.");

            const data = await response.json();

            if (data.response) {

                const newMessage: message = {
                    content: data.response,
                    role: "assistant",
                    id: uuidv4(),
                    pdfs: Array.isArray(data.pdfs) ? data.pdfs : data.pdfs ? [data.pdfs] : [],  // ✅ Ensure always an array
                    sources: Array.isArray(data.sources) ? data.sources : [],
                };
                console.log("New message object being added to state:", newMessage);
                // Append assistant's response to chat
                setMessages(prev => [...prev, newMessage]);
                // setMessages(prev => [...prev, { content: data.response, role: "assistant", id: uuidv4() }]);
            } else {
                throw new Error("Invalid response format");
            }
        } catch (error) {
            console.error("Error:", error);
            setMessages(prev => [...prev, {
                content: "Error: Unable to process request",
                role: "assistant",
                id: uuidv4()
            }]);
        } finally {
            setTimeout(() => setIsLoading(false), 1000);
        }
    }


    return (
        <div className="flex flex-col min-w-0 h-dvh bg-background">
            <Header/>

            <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4" ref={messagesContainerRef}>
                {messages.length == 0 && <Overview/>}
                {messages.map((message, index) => (
                    <PreviewMessage key={index} message={message}/>
                ))}
                {isLoading && <ThinkingMessage/>}
                {isLoading && messages.some(msg => msg.pdfs?.length > 0) && (
                    <div className="mt-4 space-y-4 animate-pulse bg-gray-200 rounded-lg h-12 w-full"></div>
                )}
                <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]"/>
            </div>
            <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
                <ChatInput
                    question={question}
                    setQuestion={setQuestion}
                    onSubmit={handleSubmit}
                    isLoading={isLoading}
                    hasApiKey={!!apiKey}
                />
            </div>
        </div>
    );
};