// lib/firebase.ts

import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
    apiKey: "AIzaSyBmPMq7v0C36y3gaIQzi5luZ-C4vkL4Z-0",
    authDomain: "princeton-f298f.firebaseapp.com",
    projectId: "princeton-f298f",
    storageBucket: "princeton-f298f.firebasestorage.app",
    messagingSenderId: "316137261516",
    appId: "1:316137261516:web:cadbf44935733eee51b569",
    measurementId: "G-3LFDJ7KFNX"
  };
// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication
const auth = getAuth(app);

// Set up Google Auth provider
const provider = new GoogleAuthProvider();

export { auth, provider };
