import "~/styles/globals.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { DemoProvider } from "~/state/DemoContext";
import Navbar from "./Navbar";

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
      <DemoProvider>
        <html lang="en" className={`${GeistSans.variable}`}>
          <body className="h-screen bg-zinc-900 text-white">
            <Navbar />
            {children}
          </body>
        </html>
      </DemoProvider>
    </ClerkProvider>
  );
}
