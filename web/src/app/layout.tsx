import "~/styles/globals.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";
import {
  ClerkProvider,
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from "@clerk/nextjs";

export const metadata: Metadata = {
  title: "scraply",
  description: "",
  icons: [{ rel: "icon", url: "/favicon.png" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <ClerkProvider>
      <html lang="en" className={`${GeistSans.variable}`}>
        <body className="h-screen bg-zinc-900 text-white">
          <SignedOut>
            <div className="flex justify-between bg-zinc-800 text-white">
              <div className="flex">
                <img src="favicon.png" className="my-auto ml-6 h-8" alt="" />
                <div className="mx-4 py-4 pr-7 font-semibold">scraply</div>
              </div>

              <SignInButton>
                <button className="mx-2 my-2 rounded-lg bg-blue-600 px-6 py-2">
                  Sign In
                </button>
              </SignInButton>
            </div>
          </SignedOut>
          <SignedIn>
            <div className="flex justify-between bg-zinc-800 text-white">
              <div className="flex">
                <img src="favicon.png" className="my-auto ml-6 h-8" alt="" />
                <div className="mx-4 py-4 pr-7 font-semibold">scraply</div>
              </div>
              <div className="my-auto px-2">
                <UserButton />
              </div>
            </div>
          </SignedIn>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
