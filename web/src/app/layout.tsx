import "~/styles/globals.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";
import Navbar from "./(components)/Navbar";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "scraply",
  description: "",
  icons: [{ rel: "icon", url: "favicon.png" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${GeistSans.variable}`}>
      <body className="h-screen bg-zinc-900 text-white">
        <Providers>
          <Navbar />
          {children}
        </Providers>
      </body>
    </html>
  );
}
