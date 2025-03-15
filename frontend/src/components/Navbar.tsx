// Navbar.tsx
import React, { useState, useEffect } from "react";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  User,
} from "firebase/auth";

// Firebase configuration import
import { firebaseApp } from "../../firebaseConfig";

const Navbar: React.FC = () => {
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isSignUp, setIsSignUp] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState("");

  const auth = getAuth(firebaseApp);

  // Monitor authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });

    // Cleanup subscription on unmount
    return () => unsubscribe();
  }, [auth]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      setShowAuthModal(false);
      setEmail("");
      setPassword("");
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (err: any) {
      console.error("Logout error:", err.message);
    }
  };

  return (
    <>
      <nav className="fixed top-0 left-0 w-full z-50 bg-gradient-to-r from-white to-gray-50 border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          {/* LEFT: Empty space for balance */}
          <div className="flex-1"></div>

          {/* CENTER: Brand with improved styling */}
          <div className="flex-1 flex justify-center">
            <h1 className="font-extrabold text-2xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600">
              ALETHEIAAI
            </h1>
          </div>

          {/* RIGHT: Sign Up/Profile Button with improved positioning */}
          <div className="flex-1 flex justify-end">
            {user ? (
              <div className="flex items-center space-x-3 bg-gray-50 px-4 py-2 rounded-lg border border-gray-200 shadow-sm">
                <span className="font-medium text-gray-800">
                  {user.email?.split("@")[0]}
                </span>
                <button
                  onClick={handleLogout}
                  className="text-sm text-gray-500 hover:text-black transition-colors"
                >
                  Logout
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="px-5 py-2 rounded-lg font-medium text-white bg-gradient-to-r from-gray-800 to-black hover:from-black hover:to-gray-700 shadow-md transition-all duration-300 transform hover:-translate-y-0.5"
              >
                Sign Up
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Auth Modal with improved styling */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50 flex items-center justify-center p-4 transition-all">
          <div className="bg-white rounded-xl w-full max-w-md p-6 shadow-2xl transform transition-all">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800">
                {isSignUp ? "Join ALETHEIAAI" : "Welcome Back"}
              </h2>
              <button
                onClick={() => setShowAuthModal(false)}
                className="text-gray-400 hover:text-gray-600 text-xl transition-colors"
              >
                &times;
              </button>
            </div>

            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded mb-6">
                <p className="text-sm">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="block text-gray-700 mb-2 font-medium">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all"
                  required
                />
              </div>

              <div>
                <label className="block text-gray-700 mb-2 font-medium">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent transition-all"
                  required
                />
              </div>

              <button
                type="submit"
                className="w-full bg-gradient-to-r from-gray-800 to-black text-white py-3 rounded-lg font-medium hover:from-black hover:to-gray-700 transition-all shadow-md mt-4"
              >
                {isSignUp ? "Create Account" : "Sign In"}
              </button>
            </form>

            <div className="mt-6 text-center">
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  setIsSignUp(!isSignUp);
                }}
                className="text-gray-600 hover:text-black transition-colors font-medium underline"
              >
                {isSignUp
                  ? "Already have an account? Sign In"
                  : "Need an account? Sign Up"}
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Navbar;
