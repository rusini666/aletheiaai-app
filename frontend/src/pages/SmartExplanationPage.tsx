import React from "react";
import { useLocation } from "react-router-dom";

interface ClassificationResult {
  prediction: string;
  explanation: string;
}

const SmartExplanationPage: React.FC = () => {
  const location = useLocation();
  const stateData = location.state as {
    classificationResult?: ClassificationResult;
  };

  // fallback if user came directly
  const predictedLabel = stateData?.classificationResult?.prediction ?? "N/A";
  const explanationText =
    stateData?.classificationResult?.explanation ?? "Explanation missing";

  // Split explanations for better display
  const explanations = explanationText.split("\n\n");

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Main content */}
      <div className="flex-grow flex flex-col items-center px-4 py-8">
        <div className="w-full max-w-3xl bg-white rounded shadow-md p-8">
          <h1 className="text-3xl font-bold text-black text-center mb-4">
            Smart AI Explanation
          </h1>

          <p className="text-xl text-gray-700 text-center mb-6">
            Predicted Label:{" "}
            <span className="font-semibold">{predictedLabel}</span>
          </p>

          <div className="space-y-6">
            {/* Explanations */}
            {explanations.map((exp, i) => (
              <div key={i} className="border rounded p-4">
                <h2 className="text-lg text-black font-semibold mb-2">
                  {exp.split(":\n")[0]}
                </h2>
                <p className="text-gray-800 whitespace-pre-wrap">
                  {exp.split(":\n")[1] || exp}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SmartExplanationPage;