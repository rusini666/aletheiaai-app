// Navbar.tsx
import React, { useState, useEffect } from "react";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  sendPasswordResetEmail,
  User
} from "firebase/auth";

import { FirebaseError } from "firebase/app";
import { firebaseApp } from "../../firebaseConfig";

const Navbar: React.FC = () => {
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isSignUp, setIsSignUp] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [resetEmail, setResetEmail] = useState("");
  const [resetSent, setResetSent] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const auth = getAuth(firebaseApp);

  // Monitor authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });

    // Cleanup subscription on unmount
    return () => unsubscribe();
  }, [auth]);

  // Reset all form states when toggling between sign-in, sign-up, and forgot password
  useEffect(() => {
    // Clear error and success messages when changing views
    setError("");
    setSuccessMessage("");
    setResetSent(false);
  }, [isSignUp, showForgotPassword]);

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

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccessMessage("");
    
    try {
      await sendPasswordResetEmail(auth, resetEmail);
      setResetSent(true);
      setSuccessMessage("Password reset email sent. Please check your inbox.");
    } catch (err: any) {
      // Handle Firebase errors
      const firebaseError = err as FirebaseError;
      if (firebaseError.code === 'auth/user-not-found') {
        setError("No account exists with this email address. Please check the email or create a new account.");
      } else {
        setError(firebaseError.message || "Failed to send reset email");
      }
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (err: any) {
      console.error("Logout error:", err.message);
    }
  };

  const resetAuthModal = () => {
    setShowAuthModal(false);
    setShowForgotPassword(false);
    setEmail("");
    setPassword("");
    setResetEmail("");
    setResetSent(false);
    setError("");
    setSuccessMessage("");
  };

  // Handle navigation to forgot password
  const goToForgotPassword = (e: React.MouseEvent) => {
    e.preventDefault();
    setError("");
    setSuccessMessage("");
    setResetSent(false);
    setShowForgotPassword(true);
    // Pre-fill reset email with sign-in email if available
    if (email) {
      setResetEmail(email);
    }
  };

  // Handle navigation back to sign in
  const goBackToSignIn = (e: React.MouseEvent) => {
    e.preventDefault();
    setError("");
    setSuccessMessage("");
    setResetSent(false);
    setShowForgotPassword(false);
  };

  return (
    <>
      <nav className="fixed top-0 left-0 w-full z-50 bg-white border-b border-indigo-100 px-6 py-4 shadow-sm">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          {/* LEFT: Empty space for balance */}
          <div className="flex-1"></div>

          {/* CENTER: Brand with updated styling */}
          <div className="flex-1 flex justify-center">
            <h1 className="font-extrabold text-2xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-700 to-blue-600">
              ALETHEIAAI
            </h1>
          </div>

          {/* RIGHT: Sign Up/Profile Button with updated styling */}
          <div className="flex-1 flex justify-end">
            {user ? (
              <div className="flex items-center space-x-3 bg-indigo-50 px-4 py-2 rounded-lg border border-indigo-100 shadow-sm">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-indigo-100 text-indigo-700 font-medium">
                  {user.email?.charAt(0).toUpperCase()}
                </div>
                <span className="font-medium text-indigo-800">
                  {user.email?.split("@")[0]}
                </span>
                <button
                  onClick={handleLogout}
                  className="text-sm text-indigo-500 hover:text-indigo-700 transition-colors"
                >
                  Logout
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="px-5 py-2 rounded-lg font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 shadow-md transition-all duration-300 transform hover:-translate-y-0.5"
              >
                Sign Up
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Auth Modal with updated styling */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-indigo-900 bg-opacity-30 backdrop-blur-sm z-50 flex items-center justify-center p-4 transition-all">
          <div className="bg-white rounded-xl w-full max-w-md p-8 shadow-2xl transform transition-all border border-indigo-100 relative">
            {/* Improved close button positioning and styling */}
            <button
              onClick={resetAuthModal}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full p-2 transition-colors"
              aria-label="Close"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                ></path>
              </svg>
            </button>

            {showForgotPassword ? (
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-indigo-900 mb-2">
                    Reset Password
                  </h2>
                  <p className="text-gray-600">
                    Enter your email address and we'll send you a link to reset your password.
                  </p>
                </div>

                {error && (
                  <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded mb-6">
                    <p className="text-sm">{error}</p>
                  </div>
                )}

                {successMessage && (
                  <div className="bg-green-50 border-l-4 border-green-500 text-green-700 p-4 rounded mb-6">
                    <p className="text-sm">{successMessage}</p>
                  </div>
                )}

                {!resetSent ? (
                  <form onSubmit={handleForgotPassword} className="space-y-5">
                    <div>
                      <label className="block text-slate-700 mb-2 font-medium">
                        Email
                      </label>
                      <input
                        type="email"
                        value={resetEmail}
                        onChange={(e) => setResetEmail(e.target.value)}
                        className="w-full px-4 py-3 border border-indigo-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                        required
                      />
                    </div>

                    <button
                      type="submit"
                      className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-indigo-700 hover:to-blue-700 transition-all shadow-md mt-4"
                    >
                      Send Reset Link
                    </button>
                  </form>
                ) : (
                  <button
                    onClick={goBackToSignIn}
                    className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-indigo-700 hover:to-blue-700 transition-all shadow-md mt-4"
                  >
                    Back to Sign In
                  </button>
                )}

                <div className="mt-6 text-center">
                  <a
                    href="#"
                    onClick={goBackToSignIn}
                    className="text-indigo-600 hover:text-indigo-800 transition-colors font-medium"
                  >
                    Return to Sign In
                  </a>
                </div>
              </>
            ) : (
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-indigo-900">
                    {isSignUp ? "Join AletheiaAI" : "Welcome Back"}
                  </h2>
                  <p className="text-gray-600 mt-1">
                    {isSignUp 
                      ? "Create an account to get started" 
                      : "Sign in to continue to your account"}
                  </p>
                </div>

                {error && (
                  <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded mb-6">
                    <p className="text-sm">{error}</p>
                  </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                  <div>
                    <label className="block text-slate-700 mb-2 font-medium">
                      Email
                    </label>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full px-4 py-3 border border-indigo-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                      required
                    />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="block text-slate-700 font-medium">
                        Password
                      </label>
                      {!isSignUp && (
                        <a
                          href="#"
                          onClick={goToForgotPassword}
                          className="text-sm text-indigo-600 hover:text-indigo-800 transition-colors"
                        >
                          Forgot password?
                        </a>
                      )}
                    </div>
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="w-full px-4 py-3 border border-indigo-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                      required
                    />
                  </div>

                  <button
                    type="submit"
                    className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-indigo-700 hover:to-blue-700 transition-all shadow-md mt-4"
                  >
                    {isSignUp ? "Create Account" : "Sign In"}
                  </button>
                </form>

                <div className="mt-6 text-center">
                  <p className="text-gray-600">
                    {isSignUp ? "Already have an account?" : "Don't have an account?"}
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setIsSignUp(!isSignUp);
                      }}
                      className="text-indigo-600 hover:text-indigo-800 transition-colors font-medium ml-1"
                    >
                      {isSignUp ? "Sign In" : "Sign Up"}
                    </a>
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default Navbar;