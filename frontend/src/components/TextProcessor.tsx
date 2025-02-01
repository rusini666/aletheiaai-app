import React, { useState, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

// 1) Use the legacy build
import * as pdfjs from "pdfjs-dist/legacy/build/pdf";

// 2) Point the workerSrc to a known CDN
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

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const navigate = useNavigate();

  // Adjust to match your backend location
  const BACKEND_URL = "http://localhost:5000";

  // --------------------------
  // 1) File Upload
  // --------------------------
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
    console.log(
      "[DEBUG] Uploaded file:",
      file.name,
      "extension:",
      fileExtension
    );

    if (fileExtension === "txt") {
      if (file.type !== "text/plain") {
        setError("Only plain text files are supported for .txt extension.");
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result;
        if (typeof text === "string") {
          console.log(
            "[DEBUG] Extracted text from TXT file:",
            text.slice(0, 100),
            "..."
          );
          setInputText(text);
          setUploadedFileName(file.name);
          setFileType("txt");
        }
      };
      reader.onerror = () => {
        setError("Failed to read the text file.");
      };
      reader.readAsText(file);
    } else if (fileExtension === "pdf") {
      if (file.type !== "application/pdf") {
        setError("Only PDF files are supported for .pdf extension.");
        return;
      }

      try {
        const arrayBuffer = await file.arrayBuffer();
        // 3) Use pdfjs (not pdfjsLib)
        const pdfDoc = await pdfjs.getDocument({ data: arrayBuffer }).promise;
        console.log("[DEBUG] PDF loaded. Number of pages:", pdfDoc.numPages);

        let textContent = "";

        for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
          const page = await pdfDoc.getPage(pageNum);
          const content = await page.getTextContent();
          console.log(`[DEBUG] Page ${pageNum} text items:`, content.items);

          const pageText = content.items.map((item: any) => item.str).join(" ");
          textContent += pageText + "\n";
        }

        console.log("[DEBUG] Final extracted text length:", textContent.length);

        if (textContent.trim() === "") {
          setError("No extractable text found in the PDF file.");
          return;
        }

        setInputText(textContent);
        setUploadedFileName(file.name);
        setFileType("pdf");
      } catch (err) {
        console.error("[DEBUG] PDF extraction error:", err);
        setError("Failed to extract text from the PDF file.");
      }
    } else {
      setError("Unsupported file type. Please upload a .txt or .pdf file.");
    }
  };

  // --------------------------
  // 2) "Detailed Explanation"
  //    (short free-text explanation from /api/classify)
  // --------------------------
  const handleDetailedExplanation = async () => {
    if (!inputText) {
      setError("Please input text or upload a valid file.");
      return;
    }

    try {
      setError("");
      const response = await axios.post(`${BACKEND_URL}/api/classify`, {
        text: inputText,
      });

      if (response.status === 200 && response.data) {
        // e.g.: { prediction, explanation }
        navigate("/smart-explanation", {
          state: {
            classificationResult: {
              prediction: response.data.prediction,
              explanation: response.data.explanation,
            } as ClassificationResult,
          },
        });
      } else {
        setError("Failed to get a detailed explanation. Please try again.");
      }
    } catch (error) {
      console.error("[DEBUG] Explanation error:", error);
      setError("An error occurred while getting the explanations.");
    }
  };

  // --------------------------
  // 3) "SHAP + LIME Explanation"
  //    (full HTML from /api/explain)
  // --------------------------
  const handleShapLimeExplanation = async () => {
    if (!inputText) {
      setError("Please input text or upload a valid file.");
      return;
    }

    try {
      setError("");
      const response = await axios.post(`${BACKEND_URL}/api/explain`, {
        text: inputText,
      });

      if (response.status === 200 && response.data) {
        navigate("/explanation-report", {
          state: { htmlContent: response.data },
        });
      } else {
        setError(
          "Failed to generate SHAP + LIME explanation. Please try again."
        );
      }
    } catch (error) {
      console.error("[DEBUG] SHAP+LIME error:", error);
      setError(
        "An error occurred while generating the SHAP + LIME explanation."
      );
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="w-full max-w-4xl p-10 bg-white rounded-lg shadow-lg mx-4">
        <h1 className="text-4xl font-extrabold text-center mb-6 text-gray-800">
          Text Analysis Tool
        </h1>
        <p className="text-lg text-center text-gray-700 mb-8">
          Please input text in the box below or upload a <strong>.txt</strong>{" "}
          or <strong>.pdf</strong> file.
        </p>

        {/* Text Input */}
        <textarea
          className="w-full h-48 p-4 border border-gray-300 rounded-lg mb-6 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
          placeholder="Enter your text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        ></textarea>

        {/* Three Buttons */}
        <div className="flex flex-wrap justify-center gap-4">
          {/* Button 1: "Detailed Explanation" */}
          <button
            onClick={handleDetailedExplanation}
            className="px-6 py-3 bg-green-500 text-white text-lg font-semibold rounded-lg hover:bg-green-600 transition"
          >
            Smart AI Explanation
          </button>

          {/* Button 2: "SHAP + LIME Explanation" */}
          <button
            onClick={handleShapLimeExplanation}
            className="px-6 py-3 bg-blue-500 text-white text-lg font-semibold rounded-lg hover:bg-blue-600 transition"
          >
            Detailed Insights Report
          </button>

          {/* Button 3: "Upload File" */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-3 bg-gray-500 text-white text-lg font-semibold rounded-lg hover:bg-gray-600 transition flex items-center gap-2"
          >
            {/* SVG Icon */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="currentColor"
              viewBox="0 0 24 24"
              className="w-6 h-6 mb-2"
            >
              <path d="M8.71,7.71,11,5.41V15a1,1,0,0,0,2,0V5.41l2.29,2.3a1,1,0,0,0,1.42,0,1,1,0,0,0,0-1.42l-4-4a1,1,0,0,0-.33-.21,1,1,0,0,0-.76,0,1,1,0,0,0-.33.21l-4,4A1,1,0,1,0,8.71,7.71ZM21,12a1,1,0,0,0-1,1v6a1,1,0,0,1-1,1H5a1,1,0,0,1-1-1V13a1,1,0,0,0-2,0v6a3,3,0,0,0,3,3H19a3,3,0,0,0,3-3V13A1,1,0,0,0,21,12Z"></path>
            </svg>
            Upload File
          </button>
        </div>

        {/* Hidden file input */}
        <input
          type="file"
          accept=".txt, .pdf"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
        />

        {/* Show file name if present */}
        {uploadedFileName && (
          <div className="text-gray-700 text-center mt-6">
            <span className="font-semibold">Uploaded File:</span>{" "}
            {uploadedFileName} ({fileType.toUpperCase()})
          </div>
        )}

        {/* Error Message */}
        {error && <div className="text-red-500 text-center mt-6">{error}</div>}
      </div>
    </div>
  );
};

export default TextProcessor;