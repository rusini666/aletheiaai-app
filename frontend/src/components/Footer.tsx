// Footer.tsx
import React, { FC } from "react";

const Footer: FC = () => {
  return (
    <footer className="bg-white border-t border-gray-300 px-6 py-4 text-black text-sm shadow-md">
      <div className="flex justify-center items-center max-w-4xl mx-auto">
        {/* CENTER: Brand Name */}
        <div className="font-extrabold text-lg">ALETHEIAAI</div>
      </div>

      {/* Horizontal Line */}
      <div className="border-t border-black my-2 w-full"></div>

      {/* Bottom Section: Copyright, Privacy & Terms */}
      <div className="text-center text-gray-600 text-xs">
        © 2024 – 2025 &nbsp;{" "}
        <a href="#" className="hover:underline">
          Privacy
        </a>{" "}
        &nbsp;–&nbsp;{" "}
        <a href="#" className="hover:underline">
          Terms
        </a>
      </div>
    </footer>
  );
};

export default Footer;