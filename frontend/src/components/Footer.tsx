import React, { FC } from "react";

const Footer: FC = () => {
  return (
    <footer className="bg-white border-t border-indigo-100 px-6 py-4 text-black text-sm shadow-sm">
      <div className="max-w-6xl mx-auto flex flex-col items-center">
        {/* CENTER: Brand Name with Gradient */}
        <div className="mb-3">
          <h1 className="font-extrabold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-700 to-blue-600">
            ALETHEIAAI
          </h1>
        </div>

        {/* Horizontal Line */}
        <div className="border-t border-indigo-100 my-2 w-full"></div>

        {/* Bottom Section: Copyright, Privacy & Terms */}
        <div className="text-center text-gray-600 text-xs flex items-center space-x-2">
          <span>© 2024 – 2025</span>
          <span className="text-indigo-500">•</span>
          <a
            href="#"
            className="text-indigo-600 hover:text-indigo-800 transition-colors"
          >
            Privacy
          </a>
          <span className="text-indigo-500">•</span>
          <a
            href="#"
            className="text-indigo-600 hover:text-indigo-800 transition-colors"
          >
            Terms
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;