"use client";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "~/util/queryClient";
import { DemoProvider } from "~/state/DemoContext";

interface ProvidersProps {
  children: React.ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <DemoProvider>{children}</DemoProvider>
    </QueryClientProvider>
  );
}
