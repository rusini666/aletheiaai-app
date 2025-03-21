import React from "react";
import { useLocation } from "react-router-dom";

interface ClassificationResult {
  prediction: string;
  explanation: string;
  probability?: number; // Add probability
  originalText?: string; // Add original text
}

const SmartExplanationPage: React.FC = () => {
  const location = useLocation();
  const stateData = location.state as {
    classificationResult?: ClassificationResult;
    originalText?: string; // Original text might be passed separately
  };

  // Extract data with fallbacks
  const predictedLabel = stateData?.classificationResult?.prediction ?? "N/A";
  const explanationText =
    stateData?.classificationResult?.explanation ?? "Explanation missing";
  const originalText =
    stateData?.originalText ||
    stateData?.classificationResult?.originalText ||
    "No input text available";

  // Calculate probabilities (ensure they're available or default to 50/50)
  const probability = stateData?.classificationResult?.probability || 0.5;
  const humanProb = predictedLabel.toLowerCase().includes("human")
    ? probability
    : 1 - probability;
  const aiProb = predictedLabel.toLowerCase().includes("ai")
    ? probability
    : 1 - probability;

  // Format probabilities as percentages
  const humanProbFormatted = (humanProb * 100).toFixed(1);
  const aiProbFormatted = (aiProb * 100).toFixed(1);

  // Split explanations for better display
  const explanations = explanationText
    .split("\n\n")
    .filter((exp) => exp.trim().length > 0);

  // Determine label class for styling
  const labelClass = predictedLabel.toLowerCase().includes("ai")
    ? "bg-indigo-600"
    : "bg-green-700";

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex flex-col">
      {/* Main content */}
      <div className="flex-grow flex flex-col items-center px-4 py-8">
        <div className="w-full max-w-3xl bg-white rounded-xl shadow-md p-8">
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-700 to-blue-500 text-center mb-6">
            Smart AI Explanation
          </h1>

          {/* Input Text Box */}
          <div className="bg-gray-50 border border-indigo-100 rounded-lg p-5 mb-6">
            <h2 className="text-xl font-bold text-indigo-700 mb-3">
              Input Text
            </h2>
            <div className="max-h-[300px] overflow-y-auto leading-relaxed">
              {originalText}
            </div>
          </div>

          {/* Classification Result */}
          <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-5 mb-6">
            <h2 className="text-xl font-bold text-indigo-700 mb-3">
              Classification Result
            </h2>

            <div
              className={`inline-block px-3 py-1 rounded-md text-white font-semibold mb-3 ${labelClass}`}
            >
              {predictedLabel}
            </div>

            {/* Probability Bar */}
            <div className="flex h-8 rounded-md overflow-hidden shadow-sm">
              <div
                className="bg-gradient-to-r from-green-400 to-green-600 flex items-center justify-center text-white font-semibold"
                style={{ width: `${humanProbFormatted}%` }}
              >
                Human: {humanProbFormatted}%
              </div>
              <div
                className="bg-gradient-to-r from-indigo-400 to-indigo-600 flex items-center justify-center text-white font-semibold"
                style={{ width: `${aiProbFormatted}%` }}
              >
                AI: {aiProbFormatted}%
              </div>
            </div>
          </div>

          {/* Explanations */}
          {explanations.length > 0 && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
              <h2 className="text-xl font-bold text-indigo-700 mb-3">
                AI-Generated Explanations
              </h2>
              <div className="space-y-4">
                {explanations.map((exp, i) => {
                  const [title, content] = exp.includes(":\n")
                    ? [exp.split(":\n")[0], exp.split(":\n")[1]]
                    : [
                        exp.includes(":")
                          ? exp.split(":")[0]
                          : `Explanation #${i + 1}`,
                        exp.includes(":")
                          ? exp.split(":").slice(1).join(":")
                          : exp,
                      ];

                  return (
                    <div
                      key={i}
                      className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm"
                    >
                      <h3 className="text-lg text-indigo-600 font-semibold mb-2">
                        {title}
                      </h3>
                      <p className="text-gray-800 whitespace-pre-wrap">
                        {content}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SmartExplanationPage;
