import React, { useState, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc =
  "https://unpkg.com/pdfjs-dist@3.5.141/legacy/build/pdf.worker.min.js";

interface ClassificationResult {
  prediction: string;
  explanation: string;
}

const TextProcessor: React.FC = () => {
  const [inputText, setInputText] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const [fileType, setFileType] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const navigate = useNavigate();
  const BACKEND_URL = "http://localhost:5000";

  // Constants for validation
  const MIN_CHAR_LIMIT = 500;
  const MAX_FILE_SIZE_MB = 5;

  // Scroll to error message
  const scrollToError = () => {
    setTimeout(() => {
      document
        .getElementById("error-message")
        ?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  // Handles file upload (TXT & PDF)
  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setUploadedFileName("");
    setInputText("");
    setFileType("");

    const fileExtension = file.name.split(".").pop()?.toLowerCase();

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File size exceeds ${MAX_FILE_SIZE_MB}MB limit.`);
      scrollToError();
      return;
    }

    if (fileExtension === "txt") {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (text.trim().length < MIN_CHAR_LIMIT) {
          setError(
            `File content must have at least ${MIN_CHAR_LIMIT} characters.`
          );
          scrollToError();
          return;
        }
        setInputText(text);
        setUploadedFileName(file.name);
        setFileType("txt");
      };
      reader.readAsText(file);
    } else if (fileExtension === "pdf") {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const pdfDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;

        let textContent = "";
        for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
          const page = await pdfDoc.getPage(pageNum);
          const content = await page.getTextContent();
          textContent +=
            content.items.map((item: any) => item.str).join(" ") + "\n";
        }

        if (textContent.trim().length < MIN_CHAR_LIMIT) {
          setError(
            `Extracted text must have at least ${MIN_CHAR_LIMIT} characters.`
          );
          scrollToError();
          return;
        }

        setInputText(textContent);
        setUploadedFileName(file.name);
        setFileType("pdf");
      } catch {
        setError("Failed to extract text from the PDF.");
        scrollToError();
      }
    } else {
      setError("Unsupported file type. Please upload a .txt or .pdf file.");
      scrollToError();
    }
  };

  // Validates if the input is acceptable
  const isValidInput = inputText.trim().length >= MIN_CHAR_LIMIT;

  // Request Smart AI Explanation
  const handleDetailedExplanation = async () => {
    if (!isValidInput) {
      setError(`Input must have at least ${MIN_CHAR_LIMIT} characters.`);
      scrollToError();
      return;
    }

    try {
      setError("");
      setIsLoading(true);
      const response = await axios.post(`${BACKEND_URL}/api/classify`, {
        text: inputText,
      });
      setIsLoading(false);
      if (response.status === 200) {
        navigate("/smart-explanation", {
          state: {
            classificationResult: response.data as ClassificationResult,
          },
        });
      } else {
        setError("Failed to get a detailed explanation.");
        scrollToError();
      }
    } catch {
      setError("An error occurred while getting the explanations.");
      setIsLoading(false);
      scrollToError();
    }
  };

  // Request SHAP + LIME Explanation
  const handleShapLimeExplanation = async () => {
    if (!isValidInput) {
      setError(`Input must have at least ${MIN_CHAR_LIMIT} characters.`);
      scrollToError();
      return;
    }

    try {
      setError("");
      setIsLoading(true);
      const response = await axios.post(`${BACKEND_URL}/api/explain`, {
        text: inputText,
      });
      setIsLoading(false);
      if (response.status === 200) {
        navigate("/explanation-report", {
          state: { htmlContent: response.data },
        });
      } else {
        setError("Failed to generate SHAP + LIME explanation.");
        scrollToError();
      }
    } catch {
      setError("An error occurred while generating the explanation.");
      setIsLoading(false);
      scrollToError();
    }
  };

  return (
    <div className="bg-white flex items-center justify-center py-8">
      <div className="bg-gray-100 w-full max-w-3xl p-6 rounded-lg shadow-lg">
        <h1 className="text-2xl font-extrabold text-center text-gray-800">
          Text Analysis Tool
        </h1>
        <p className="text-md text-center text-gray-600 mt-1">
          Input text or upload a <span className="font-bold">.txt</span> or{" "}
          <span className="font-bold">.pdf</span> file.
        </p>

        <textarea
          className="w-full h-64 p-3 border border-gray-300 rounded-lg mt-4 
             focus:ring-2 focus:ring-blue-500 text-md resize-none overflow-auto"
          placeholder="Enter your text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />

        <div className="flex flex-wrap justify-center gap-4 mt-4">
          <button
            onClick={handleDetailedExplanation}
            className={`px-5 py-2 font-semibold rounded-lg transition shadow-md ${
              isValidInput
                ? "bg-green-500 text-white hover:bg-green-600"
                : "bg-gray-300 text-gray-600 cursor-not-allowed"
            }`}
            disabled={!isValidInput || isLoading}
          >
            {isLoading ? "Processing..." : "Smart AI Explanation"}
          </button>

          <button
            onClick={handleShapLimeExplanation}
            className={`px-5 py-2 font-semibold rounded-lg transition shadow-md ${
              isValidInput
                ? "bg-blue-500 text-white hover:bg-blue-600"
                : "bg-gray-300 text-gray-600 cursor-not-allowed"
            }`}
            disabled={!isValidInput || isLoading}
          >
            {isLoading ? "Processing..." : "Detailed Insights Report"}
          </button>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-5 py-2 bg-gray-700 text-white font-semibold rounded-lg 
                       hover:bg-gray-800 transition shadow-md"
          >
            Upload File
          </button>
        </div>

        <input
          type="file"
          accept=".txt,application/pdf"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
        />

        {uploadedFileName && (
          <div className="mt-3 text-center text-gray-700 text-md">
            <span className="font-semibold">Uploaded File:</span>{" "}
            {uploadedFileName} ({fileType.toUpperCase()})
          </div>
        )}

        {error && (
          <div
            id="error-message"
            className="mt-3 text-center text-red-500 text-md font-semibold"
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default TextProcessor;