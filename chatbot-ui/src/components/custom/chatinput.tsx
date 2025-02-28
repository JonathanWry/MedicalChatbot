import { Textarea } from "../ui/textarea";
import { cx } from 'classix';
import { Button } from "../ui/button";
import {ArrowUpIcon, ChevronDownIcon} from "./icons"
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import {useRef, useState} from 'react';

interface ChatInputProps {
    question: string;
    setQuestion: (question: string) => void;
    onSubmit: (text?: string, mode?:string) => void;
    isLoading: boolean;
    hasApiKey: boolean;
}

const suggestedActions = [
    {
        title: 'Give me information regarding',
        label: 'tumor inhibiting properties',
        action: 'Give me information regarding tumor inhibiting properties',
    },
    {
        title: 'Tell me information',
        label: 'about o-salicylate',
        action: 'Tell me information about o-salicylate',
    },
];
const methodOptions = [
    { label: "Direct text Summarization", value: "direct" },
    { label: "KG Aid 1", value: "kg1" },
    { label: "KG Aid 2", value: "kg2" },
];

export const ChatInput = ({ question, setQuestion, onSubmit, isLoading,hasApiKey }: ChatInputProps) => {
    const [showSuggestions, setShowSuggestions] = useState(true);
    const [isMethodOpen, setIsMethodOpen] = useState(false);
    const [selectedMethod, setSelectedMethod] = useState(methodOptions[0]);
    const methodRef = useRef(null);

    const handleMethodSelect = (method: { label: string, value: string }) => {
        console.log("Method selected:", method);  // Debugging output
        setSelectedMethod((prev) => {
            console.log("Updating mode to:", method.value);
            return method;
        });
        setIsMethodOpen(false);
    };

    return(
    <div className="relative w-full flex flex-col gap-4">
        {showSuggestions && (
            <div className="hidden md:grid sm:grid-cols-2 gap-2 w-full">
                {suggestedActions.map((suggestedAction, index) => (
                    <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 20 }}
                    transition={{ delay: 0.05 * index }}
                    key={index}
                    className={index > 1 ? 'hidden sm:block' : 'block'}
                    >
                        <Button
                            variant="ghost"
                            onClick={ () => {
                                if (!hasApiKey) {
                                    toast.error("Please set your OpenAI API key in the sidebar first!");
                                    return;
                                }
                                const text = suggestedAction.action;
                                console.log("Submitting mode:", selectedMethod.value);
                                onSubmit(text, selectedMethod.value);
                                setShowSuggestions(false);
                            }}
                            disabled={isLoading}
                            className="text-left border rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start"
                        >
                            <span className="font-medium">{suggestedAction.title}</span>
                            <span className="text-muted-foreground">
                            {suggestedAction.label}
                            </span>
                        </Button>
                    </motion.div>
                ))}
            </div>
        )}

        <input
        type="file"
        className="fixed -top-4 -left-4 size-0.5 opacity-0 pointer-events-none"
        multiple
        tabIndex={-1}
        />
        <div className="relative w-full">

            <Textarea
            placeholder="Send a message..."
            className={cx(
                'min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-xl text-base bg-muted',
            )}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();

                    if (isLoading) {
                        toast.error('Please wait for the model to finish its response!');
                    } else {
                        console.log("Submitting via Enter mode:", selectedMethod.value);
                        setShowSuggestions(false);
                        onSubmit(question, selectedMethod.value);
                    }
                }
            }}
            rows={3}
            autoFocus
            />
            <Button
                className="rounded-full p-1.5 h-fit absolute bottom-2 left-2 m-0.5 border dark:border-zinc-600"
                onClick={() => setIsMethodOpen(!isMethodOpen)}
            >
                <span className="text-sm">{selectedMethod.label}</span>
                <ChevronDownIcon size={14} className="ml-1" />
            </Button>
            <Button
                className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
                onClick={() =>{
                    console.log("Submitting mode:", selectedMethod.value);
                    console.log("hasApiKey", hasApiKey);
                    if (!hasApiKey) {
                        toast.error("Please set your OpenAI API key in the sidebar first!");
                        return;
                    }
                    onSubmit(question, selectedMethod.value)
                    }
                }
                disabled={question.length === 0 || isLoading}
            >
                <ArrowUpIcon size={14} />
            </Button>
        </div>
        {/* Method Dropdown */}
        {isMethodOpen && (
            <div className="absolute bottom-12 left-2 bg-background border rounded-lg shadow-lg z-10">
                {methodOptions.map((method, index) => (
                    <div
                        key={index}
                        className="px-4 py-2 hover:bg-accent cursor-pointer"
                        onClick={() => handleMethodSelect(method)}
                    >
                        {method.label}
                    </div>
                ))}
            </div>
        )}
    </div>
    );
}