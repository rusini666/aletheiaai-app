// In your route config:
// <Route path="/explanation-report" element={<ExplanationReport />} />

// Then the component:
import React from "react";
import { useLocation } from "react-router-dom";

const ExplanationReport: React.FC = () => {
  const location = useLocation();
  const { htmlContent } = location.state || {};

  return <div dangerouslySetInnerHTML={{ __html: htmlContent }} />;
};

export default ExplanationReport;