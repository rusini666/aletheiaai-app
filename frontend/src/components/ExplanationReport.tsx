// ExplanationReport.tsx
import React from "react";
import { useLocation } from "react-router-dom";

const ExplanationReport: React.FC = () => {
  const location = useLocation();
  const { htmlContent } = location.state || {};

  return (
    // Just use a normal container
    <div className="w-full p-0 m-0 bg-white">
      <iframe
        srcDoc={htmlContent}
        title="SHAP-LIME-Report"
        data-testid="explanation-iframe"
        // Let the iframe height auto-adjust or set a min height
        className="w-full min-h-screen border-none bg-white"
      />
    </div>
  );
};

export default ExplanationReport;
