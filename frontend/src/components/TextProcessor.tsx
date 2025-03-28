import React, { useState, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

interface ClassificationResult {
  prediction: string;
  explanation: string;
  originalText?: string; // Added to receive this from server if available
}

const TextProcessor: React.FC = () => {
  const [inputText, setInputText] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const [fileType, setFileType] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [processingButton, setProcessingButton] = useState<string | null>(null);

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
        // Send the PDF file to the Flask backend for extraction using PyPDF2.
        const formData = new FormData();
        formData.append("file", file);
        const response = await axios.post(
          `${BACKEND_URL}/extract_pdf_text`,
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );
        const extractedText = response.data.text;
        if (!extractedText || extractedText.trim().length < MIN_CHAR_LIMIT) {
          setError(
            `Extracted text must have at least ${MIN_CHAR_LIMIT} characters.`
          );
          scrollToError();
          return;
        }
        setInputText(extractedText);
        setUploadedFileName(file.name);
        setFileType("pdf");
      } catch (e) {
        console.error("Error extracting PDF text via backend:", e);
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
      setProcessingButton("smart"); // Track which button is processing
      const response = await axios.post(`${BACKEND_URL}/api/classify`, {
        text: inputText,
      });
      setIsLoading(false);
      setProcessingButton(null);
      if (response.status === 200) {
        // IMPORTANT FIX: Always include the original text
        const result = response.data as ClassificationResult;

        navigate("/smart-explanation", {
          state: {
            classificationResult: result,
            originalText: inputText, // Explicitly include the input text
          },
        });
      } else {
        setError("Failed to get a detailed explanation.");
        scrollToError();
      }
    } catch {
      setError("An error occurred while getting the explanations.");
      setIsLoading(false);
      setProcessingButton(null);
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
      setProcessingButton("detailed"); // Track which button is processing
      const response = await axios.post(`${BACKEND_URL}/api/explain`, {
        text: inputText,
      });
      setIsLoading(false);
      setProcessingButton(null);
      if (response.status === 200) {
        navigate("/explanation-report", {
          state: {
            htmlContent: response.data,
            originalText: inputText, // Also include original text for consistency
          },
        });
      } else {
        setError("Failed to generate SHAP + LIME explanation.");
        scrollToError();
      }
    } catch {
      setError("An error occurred while generating the explanation.");
      setIsLoading(false);
      setProcessingButton(null);
      scrollToError();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex items-center justify-center py-8">
      <div className="bg-white w-full max-w-3xl p-8 rounded-xl shadow-xl border border-indigo-100">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-extrabold text-indigo-900 mb-2">
            Text Analysis Tool
          </h1>
          <p className="text-lg text-slate-600">
            Input text or upload a{" "}
            <span className="font-bold text-indigo-600">.txt</span> or{" "}
            <span className="font-bold text-indigo-600">.pdf</span> file for
            analysis
          </p>
        </div>

        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-slate-500 font-medium">
              Min. {MIN_CHAR_LIMIT} characters required
            </span>
          </div>
          <textarea
            className="w-full h-64 p-4 border border-indigo-200 rounded-lg
               focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-md 
               resize-none overflow-auto transition shadow-sm"
            placeholder="Enter your text here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
        </div>

        <div className="flex flex-wrap justify-center gap-4 mb-6">
          <button
            onClick={handleDetailedExplanation}
            className={`px-5 py-3 font-semibold rounded-lg transition-all duration-200 shadow-md flex items-center ${
              isValidInput && !processingButton
                ? "bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:shadow-lg hover:translate-y-[-2px]"
                : "bg-gray-200 text-gray-400 cursor-not-allowed"
            }`}
            disabled={!isValidInput || processingButton !== null}
          >
            {processingButton === "smart" ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </>
            ) : (
              "Smart AI Explanation"
            )}
          </button>

          <button
            onClick={handleShapLimeExplanation}
            className={`px-5 py-3 font-semibold rounded-lg transition-all duration-200 shadow-md flex items-center ${
              isValidInput && !processingButton
                ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-lg hover:translate-y-[-2px]"
                : "bg-gray-200 text-gray-400 cursor-not-allowed"
            }`}
            disabled={!isValidInput || processingButton !== null}
          >
            {processingButton === "detailed" ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </>
            ) : (
              "Detailed Insights Report"
            )}
          </button>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-5 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white font-semibold rounded-lg 
                      hover:shadow-lg hover:translate-y-[-2px] transition-all duration-200 shadow-md flex items-center"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              ></path>
            </svg>
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
          <div className="mb-4 p-3 bg-indigo-50 border border-indigo-100 rounded-lg flex items-center justify-center">
            <svg
              className="w-5 h-5 text-indigo-500 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              ></path>
            </svg>
            <span className="font-medium text-indigo-800">
              {uploadedFileName}{" "}
              <span className="text-indigo-500 font-normal">
                ({fileType.toUpperCase()})
              </span>
            </span>
          </div>
        )}

        {!uploadedFileName && !error && (
          <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-100">
            <h3 className="font-medium text-indigo-800 mb-2 flex items-center">
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                ></path>
              </svg>
              File Requirements
            </h3>
            <ul className="text-sm text-slate-600 ml-6 list-disc">
              <li>
                File types: <span className="font-semibold">.txt</span> or{" "}
                <span className="font-semibold">.pdf</span>
              </li>
              <li>
                Maximum size:{" "}
                <span className="font-semibold">{MAX_FILE_SIZE_MB}MB</span>
              </li>
              <li>
                Minimum length:{" "}
                <span className="font-semibold">
                  {MIN_CHAR_LIMIT} characters
                </span>
              </li>
            </ul>
          </div>
        )}

        {error && (
          <div
            id="error-message"
            className="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 flex items-start"
          >
            <svg
              className="w-5 h-5 mr-2 mt-0.5 flex-shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              ></path>
            </svg>
            <span>{error}</span>
          </div>
        )}

        <div className="mt-6 text-center text-slate-500 text-sm border-t border-slate-100 pt-4">
          AletheiaAI helps detect AI-generated content with user-centric
          explainability
        </div>
      </div>
    </div>
  );
};

export default TextProcessor;