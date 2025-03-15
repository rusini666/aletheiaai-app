import React, { useState } from "react";
import { auth } from "../firebaseConfig";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  User,
} from "firebase/auth";

const Auth: React.FC = () => {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string>("");

  // Sign Up Function
  const handleSignUp = async () => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      setUser(userCredential.user);
      setError("");
      console.log("User signed up:", userCredential.user);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Sign In Function
  const handleSignIn = async () => {
    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email,
        password
      );
      setUser(userCredential.user);
      setError("");
      console.log("User signed in:", userCredential.user);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Sign Out Function
  const handleSignOut = async () => {
    try {
      await signOut(auth);
      setUser(null);
      setError("");
      console.log("User signed out");
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="flex flex-col items-center bg-gray-100 p-6 rounded-lg shadow-md w-80 mx-auto mt-10">
      <h2 className="text-2xl font-bold mb-4">Firebase Auth</h2>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {!user ? (
        <>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="border p-2 rounded w-full mb-2"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="border p-2 rounded w-full mb-2"
          />

          <button
            onClick={handleSignUp}
            className="bg-blue-500 text-white p-2 rounded w-full mb-2"
          >
            Sign Up
          </button>
          <button
            onClick={handleSignIn}
            className="bg-green-500 text-white p-2 rounded w-full"
          >
            Sign In
          </button>
        </>
      ) : (
        <>
          <p className="text-green-600 font-semibold">
            Logged in as {user.email}
          </p>
          <button
            onClick={handleSignOut}
            className="bg-red-500 text-white p-2 rounded w-full mt-4"
          >
            Sign Out
          </button>
        </>
      )}
    </div>
  );
};

export default Auth;