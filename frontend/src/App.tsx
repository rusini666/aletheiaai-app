import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import TextProcessor from "./components/TextProcessor";
import ExplanationReport from "./components/ExplanationReport";
import SmartExplanationPage from "./pages/SmartExplanationPage";

function App() {
  return (
    <Router>
      <Routes>
        {/* The parent route uses Layout, which has the Navbar */}
        <Route element={<Layout />}>
          <Route path="/" element={<TextProcessor />} />
          <Route path="/explanation-report" element={<ExplanationReport />} />
          <Route path="/smart-explanation" element={<SmartExplanationPage />} />
          {/* Add more routes here if needed */}
        </Route>
      </Routes>
    </Router>
  );
}

export default App;