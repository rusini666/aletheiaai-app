// src/App.tsx
import React from "react";
import TextProcessor from "./components/TextProcessor";

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <TextProcessor />
    </div>
  );
};

export default App;