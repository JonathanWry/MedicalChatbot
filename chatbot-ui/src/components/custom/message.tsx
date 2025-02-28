import {useState, useEffect} from 'react';
import {motion, AnimatePresence} from 'framer-motion';
import {cx} from 'classix';
import {SparklesIcon} from './icons';
import {Markdown} from './markdown';
import {message} from "../../interfaces/interfaces"
import {MessageActions} from '@/components/custom/actions';
import {X} from "lucide-react";
import * as pdfjsLib from 'pdfjs-dist/build/pdf';
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.js?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;


async function generatePdfThumbnail(pdfBase64: string): Promise<string | null> {
    try {
        const pdfData = atob(pdfBase64); // decode base64 to binary
        const loadingTask = pdfjsLib.getDocument({ data: pdfData });
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(1);

        const scale = 0.5;
        const viewport = page.getViewport({ scale });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) return null;

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        // Render page into canvas
        await page.render({ canvasContext: context, viewport }).promise;

        // Return the thumbnail as base64 image
        return canvas.toDataURL('image/png');
    } catch (error) {
        console.error("Error generating thumbnail:", error);
        return null;
    }
}

export const PreviewMessage = ({message}: { message: message; }) => {
    const [selectedPdf, setSelectedPdf] = useState<string | null>(null);
    const [pdfThumbnails, setPdfThumbnails] = useState<string[]>([]);
    useEffect(() => {
        const generateThumbnails = async () => {
            if (!Array.isArray(message.pdfs)) return;
            const thumbnails = await Promise.all(
                message.pdfs.map(async (pdfBase64) => {
                    const thumb = await generatePdfThumbnail(pdfBase64);
                    return thumb;
                })
            );
            setPdfThumbnails(thumbnails.filter(Boolean) as string[]);
        };

        generateThumbnails();
    }, [message.pdfs]);
    const generateImageThumbnail = async (imageBase64: string): Promise<string> => {
        // Simulate thumbnail generation (you can use a library like `browser-image-compression` for actual compression)
        return new Promise((resolve) => {
            const img = new Image();
            img.src = imageBase64;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const maxWidth = 128; // Thumbnail width
                const scale = maxWidth / img.width;
                canvas.width = maxWidth;
                canvas.height = img.height * scale;
                const ctx = canvas.getContext('2d');
                ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL());
            };
        });
    };
    return (
        <motion.div
            className="w-full mx-auto max-w-3xl px-4 group/message"
            initial={{y: 5, opacity: 0}}
            animate={{y: 0, opacity: 1}}
            data-role={message.role}
        >
            <div
                className={cx(
                    'group-data-[role=user]/message:bg-zinc-700 dark:group-data-[role=user]/message:bg-muted group-data-[role=user]/message:text-white flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl'
                )}
            >
                {message.role === 'assistant' && (
                    <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
                        <SparklesIcon size={14}/>
                    </div>
                )}

                <div className="flex flex-col w-full">
                    {message.content && (
                        <div className="flex flex-col gap-4 text-left">
                            <Markdown>{message.content}</Markdown>
                            {/* Line break before sources */}
                            {Array.isArray(message.sources) && message.sources.length > 0 && (
                                <>
                                    <div className="text-sm italic mt-2">Sources:</div>
                                    <ul className="text-sm italic ml-4 list-disc">
                                        {message.sources.map((source, index) => (
                                            <li key={index}>{source}</li>
                                        ))}
                                    </ul>
                                </>
                            )}
                        </div>
                    )}
                    {/* PDF Thumbnails */}
                    {Array.isArray(message.pdfs) && message.pdfs.length > 0 && (
                        <div className="mt-4 flex flex-wrap gap-4">
                            {message.pdfs.map((pdfBase64, index) => (
                                <div key={`pdf-${index}`} className="flex flex-col items-center">
                                    <button
                                        className="relative border rounded-md shadow-md hover:shadow-lg transition"
                                        onClick={() => setSelectedPdf(pdfBase64)}
                                    >
                                        {/* If thumbnail is ready, display it */}
                                        {pdfThumbnails[index] ? (
                                            <img
                                                src={pdfThumbnails[index]}
                                                alt={`PDF Preview ${index + 1}`}
                                                className="w-32 h-40 object-cover rounded-md"
                                            />
                                        ) : (
                                            <div className="w-32 h-40 flex items-center justify-center bg-gray-200 dark:bg-gray-800 rounded-md">
                                                📄 Loading...
                                            </div>
                                        )}
                                    </button>
                                    <span className="text-sm mt-1">PDF {index + 1}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {message.role === 'assistant' && (
                        <MessageActions message={message}/>
                    )}

                </div>
            </div>
            {/* Modal for Full PDF View */}
            {selectedPdf && (
                <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg shadow-lg w-[90%] h-[90%] relative flex flex-col">
                        <div className="flex justify-end p-2">
                            <button
                                onClick={() => setSelectedPdf(null)}
                                className="ml-auto bg-transparent text-gray-500 hover:text-gray-800 p-2 rounded-full"
                            >
                                <X className="h-6 w-6" />
                            </button>
                        </div>
                        <div className="flex-1 px-4 pb-4">
                            <iframe
                                src={`data:application/pdf;base64,${selectedPdf}#toolbar=1&navpanes=0`}
                                className="w-full h-full border rounded-lg"
                            />
                        </div>
                    </div>
                </div>
            )}
        </motion.div>
    );
};

export const ThinkingMessage = () => {
    const role = 'assistant';

    return (
        <motion.div
            className="w-full mx-auto max-w-3xl px-4 group/message "
            initial={{y: 5, opacity: 0}}
            animate={{y: 0, opacity: 1, transition: {delay: 0.2}}}
            data-role={role}
        >
            <div
                className={cx(
                    'flex gap-4 group-data-[role=user]/message:px-3 w-full group-data-[role=user]/message:w-fit group-data-[role=user]/message:ml-auto group-data-[role=user]/message:max-w-2xl group-data-[role=user]/message:py-2 rounded-xl',
                    'group-data-[role=user]/message:bg-muted'
                )}
            >
                <div className="size-8 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border">
                    <SparklesIcon size={14}/>
                </div>
                <p className="text-gray-600 dark:text-gray-300 text-sm">Thinking...</p>
            </div>
        </motion.div>
    );
};
