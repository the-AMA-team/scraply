"use client";
import { useContext, createContext, useState, useEffect } from "react";
import { auth, provider } from "../../utils/firebase";
import { getAuth, signInWithPopup, signOut, User } from "firebase/auth";
import axios from "axios";

interface UserContextType {
  user: User | null;
  handleGoogleSignIn: () => void;
  handleGoogleSignOut: () => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => useContext(UserContext);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (user) {
        setUser(user);
      } else {
        setUser(null);
      }
    });

    return () => unsubscribe();
  }, []);

  const handleGoogleSignIn = async () => {
    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      setUser(user);
      console.log("User Info: ", user);

      const data = {
        id: user.uid,
        name: user.displayName,
      };

      try {
        const response = await axios.post(
          "http://localhost:3000/api/user",
          data
        );
        console.log(response);
      } catch (error) {
        console.error(error);
      }
    } catch (error: any) {
      console.error("Error during Google sign-in:", error.message);
    }
  };

  const handleGoogleSignOut = async () => {
    try {
      await signOut(auth);
      setUser(null);
      console.log("User signed out successfully");
    } catch (error) {
      console.error("Error signing out: ", error);
    }
  };

  return (
    <UserContext.Provider
      value={{
        user,
        handleGoogleSignIn,
        handleGoogleSignOut,
      }}
    >
      {children}
    </UserContext.Provider>
  );
};
