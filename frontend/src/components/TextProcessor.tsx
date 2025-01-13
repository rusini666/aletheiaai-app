import React, { useState, useRef } from "react";
import axios from "axios";
import * as pdfjsLib from "pdfjs-dist";

const TextProcessor: React.FC = () => {
  const [inputText, setInputText] = useState<string>("");
  const [prediction, setPrediction] = useState<string>("");
  const [explanation, setExplanation] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const [fileType, setFileType] = useState<string>("");

  // Reference to the hidden file input
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Define the backend URL (will use environment variable later)
  const BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";

  const handleFreeTextExplanation = async () => {
    setLoading(true);
    setError("");
    setPrediction("");
    setExplanation("");
    try {
      const response = await axios.post(`${BACKEND_URL}/api/classify`, {
        text: inputText,
      });
      setPrediction(response.data.prediction);
      setExplanation(response.data.explanation);
    } catch (err: any) {
      setError(
        err.response?.data?.error ||
          "An error occurred while generating the explanation."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      // Reset previous errors and outputs
      setError("");
      setPrediction("");
      setExplanation("");
      setUploadedFileName("");
      setInputText("");
      setFileType("");

      const fileExtension = file.name.split(".").pop()?.toLowerCase();

      if (fileExtension === "txt") {
        // Handle TXT files
        if (file.type !== "text/plain") {
          setError("Only plain text files are supported for .txt extension.");
          return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result;
          if (typeof text === "string") {
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
        // Handle PDF files
        if (file.type !== "application/pdf") {
          setError("Only PDF files are supported for .pdf extension.");
          return;
        }

        try {
          const arrayBuffer = await file.arrayBuffer();
          const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
          let textContent = "";

          for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            const content = await page.getTextContent();
            const pageText = content.items
              .map((item: any) => item.str)
              .join(" ");
            textContent += pageText + "\n";
          }

          if (textContent.trim() === "") {
            setError("No extractable text found in the PDF file.");
            return;
          }

          setInputText(textContent);
          setUploadedFileName(file.name);
          setFileType("pdf");
        } catch (err) {
          console.error(err);
          setError("Failed to extract text from the PDF file.");
        }
      } else {
        setError("Unsupported file type. Please upload a .txt or .pdf file.");
      }
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-4">Text Analysis Tool</h1>
      <textarea
        className="w-full h-32 p-2 border border-gray-300 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
        placeholder="Enter your text here..."
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      ></textarea>
      <div className="flex space-x-4 mb-4">
        <button
          onClick={handleFreeTextExplanation}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition"
          disabled={loading || !inputText}
        >
          {loading ? "Generating Explanation..." : "Free-Text Explanation"}
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition flex items-center"
          disabled={loading}
          title="Upload a .txt or .pdf file"
        >
          {/* Attachment SVG Icon */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 mr-2"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M10 3a5 5 0 00-4.546 2.916L3.96 7.414a4.5 4.5 0 005.096 5.095l.829-.828a1 1 0 011.414 0l3.182 3.182a1 1 0 001.414-1.414l-3.182-3.182a3 3 0 00-4.242-4.242l-.829.828A3 3 0 0010 3z"
              clipRule="evenodd"
            />
          </svg>
          Upload File
        </button>
        <input
          type="file"
          accept=".txt,application/pdf"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>
      {uploadedFileName && (
        <div className="mb-4 text-gray-700">
          <span className="font-medium">Uploaded File:</span> {uploadedFileName}{" "}
          ({fileType.toUpperCase()})
        </div>
      )}
      {error && <div className="text-red-500 mb-4">{error}</div>}
      {prediction && (
        <div className="mb-4">
          <h2 className="text-xl font-semibold mb-2">Prediction</h2>
          <p className="text-gray-700">{prediction}</p>
        </div>
      )}
      {explanation && (
        <div className="mb-4">
          <h2 className="text-xl font-semibold mb-2">Free-Text Explanation</h2>
          <p className="text-gray-700 whitespace-pre-wrap">{explanation}</p>
        </div>
      )}
    </div>
  );
};

export default TextProcessor;