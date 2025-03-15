// Layout.tsx
import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import Footer from "./Footer";

const Layout: React.FC = () => {
  return (
    // Full-page flex container
    <div className="flex flex-col min-h-screen">
      <Navbar />

      {/* Main content has top padding because Navbar is fixed */}
      <main className="flex-grow pt-[4rem]">
        <Outlet />
      </main>

      <Footer />
    </div>
  );
};

export default Layout;