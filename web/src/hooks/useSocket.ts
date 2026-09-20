import { useEffect, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";
import { API_CONFIG, SOCKET_CONFIG } from "~/util/config";

const JOB_STORAGE_KEY = "scraply_training_job_id";

function readStoredJobId(): string | null {
  try {
    return sessionStorage.getItem(JOB_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredJobId(jobId: string) {
  try {
    sessionStorage.setItem(JOB_STORAGE_KEY, jobId);
  } catch {
    // sessionStorage can throw in private browsing
  }
}

function clearStoredJobId() {
  try {
    sessionStorage.removeItem(JOB_STORAGE_KEY);
  } catch {
    // ignore
  }
}

interface TrainingProgress {
  epoch: number;
  total_epochs: number;
  progress: number;
  train_loss: number;
  train_accuracy: number;
  test_loss: number;
  test_accuracy: number;
  train_losses: { x: number; y: number }[];
  test_losses: { x: number; y: number }[];
}

interface TrainingCompleted {
  final_results: any;
  message: string;
}

interface UseSocketReturn {
  socket: Socket | null;
  isConnected: boolean;
  trainingProgress: TrainingProgress | null;
  isTrainingActive: boolean;
  isTrainingPaused: boolean;
  isTrainingPausing: boolean;
  trainingCompleted: TrainingCompleted | null;
  trainingError: string | null;
  startTraining: (config: any) => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stopTraining: () => void;
  resetTraining: () => void;
  checkTrainingStatus: () => void;
}

export const useSocket = (): UseSocketReturn => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [trainingProgress, setTrainingProgress] =
    useState<TrainingProgress | null>(null);
  const [isTrainingActive, setIsTrainingActive] = useState(false);
  const [isTrainingPaused, setIsTrainingPaused] = useState(false);
  const [isTrainingPausing, setIsTrainingPausing] = useState(false);
  const [trainingCompleted, setTrainingCompleted] =
    useState<TrainingCompleted | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const socketRef = useRef<Socket | null>(null);
  const processedCompletedResultsRef = useRef<string | null>(null);
  const isTrainingActiveRef = useRef<boolean>(false);

  useEffect(() => {
    // Create socket connection
    const newSocket = io(SOCKET_CONFIG.URL, {
      transports: ["websocket"],
      autoConnect: true,
    });

    socketRef.current = newSocket;
    setSocket(newSocket);

    let isTabClosing = false; // Track if tab is actually closing vs just hidden

    // Connection events
    newSocket.on("connect", () => {
      console.log("Connected to training server");
      setIsConnected(true);

      const jobId = readStoredJobId();
      if (jobId) {
        newSocket.emit("join_job", { job_id: jobId });
      } else {
        newSocket.emit("check_training_status");
      }
    });

    newSocket.on("disconnect", () => {
      console.log("Disconnected from training server");
      setIsConnected(false);
      setIsTrainingActive(false);
    });

    newSocket.on("job_not_found", () => {
      clearStoredJobId();
    });

    // Use Page Visibility API to detect tab switching vs closing
    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        // Tab became hidden - could be switching tabs or closing
        // Send a signal to backend that we're hidden (but don't stop training)
        if (newSocket.connected) {
          newSocket.emit("tab_hidden");
        }
      } else if (document.visibilityState === "visible") {
        // Tab became visible again - definitely just switched tabs
        isTabClosing = false;
        if (newSocket.connected) {
          newSocket.emit("tab_visible");
          const jobId = readStoredJobId();
          if (jobId) {
            newSocket.emit("join_job", { job_id: jobId });
          } else {
            newSocket.emit("check_training_status");
          }
        }
      }
    };

    // Detect when tab is actually closing (not just hidden)
    const handleBeforeUnload = () => {
      isTabClosing = true;
      if (newSocket.connected && isTrainingActiveRef.current) {
        // Try to emit stop_training before page closes
        // Use sendBeacon as fallback for more reliable delivery
        try {
          newSocket.emit("stop_training");
        } catch (e) {
          // If socket fails, try using Beacon API as fallback
          const data = JSON.stringify({ action: "stop_training" });
          navigator.sendBeacon(
            SOCKET_CONFIG.URL.replace("ws://", "http://").replace("wss://", "https://") + "/stop-on-close",
            data
          );
        }
      }
    };

    // Use pagehide event (more reliable than beforeunload for detecting closure)
    const handlePageHide = (event: PageTransitionEvent) => {
      if (event.persisted) {
        // Page is being cached (e.g., back/forward navigation) - don't stop training
        return;
      }
      // Page is being unloaded - likely closing
      if (isTrainingActiveRef.current && newSocket.connected) {
        try {
          newSocket.emit("stop_training");
        } catch (e) {
          console.log("Could not send stop_training on pagehide");
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("beforeunload", handleBeforeUnload);
    window.addEventListener("pagehide", handlePageHide);

    // Training events
    newSocket.on("training_started", (data) => {
      console.log("Training started:", data);
      if (data?.job_id) {
        writeStoredJobId(data.job_id);
      }
      isTrainingActiveRef.current = true;
      setIsTrainingActive(true);
      setIsTrainingPausing(false);
      setIsTrainingPaused(false);
      setTrainingCompleted(null);
      setTrainingError(null);
      setTrainingProgress(null);
      processedCompletedResultsRef.current = null; // Clear ref for new training
    });

    newSocket.on("epoch_started", (data: any) => {
      console.log("Epoch started:", data);
    });

    newSocket.on("epoch_completed", (data: TrainingProgress) => {
      console.log("Epoch completed:", data);
      setTrainingProgress(data);
    });

    newSocket.on("training_completed", (data: TrainingCompleted) => {
      console.log("Training completed:", data);
      // Update ref to prevent reprocessing from status checks
      const resultsHash = JSON.stringify(data.final_results?.training);
      processedCompletedResultsRef.current = resultsHash;
      isTrainingActiveRef.current = false;
      setIsTrainingPausing(false);
      setIsTrainingPaused(false);
      setTrainingCompleted(data);
      setIsTrainingActive(false);
    });

    newSocket.on("training_error", (data: any) => {
      console.error("Training error:", data);
      isTrainingActiveRef.current = false;
      setIsTrainingPausing(false);
      setIsTrainingPaused(false);
      setTrainingError(data.error);
      setIsTrainingActive(false);
    });

    // Handle training status check response
    newSocket.on("training_status", (data: any) => {
      console.log("Training status:", data);
      if (data.job_id) {
        writeStoredJobId(data.job_id);
      }
      if (data.is_training) {
        isTrainingActiveRef.current = true;
        setIsTrainingActive(true);
        setIsTrainingPaused(data.pause_confirmed || data.is_paused || false);
        setIsTrainingPausing(Boolean(data.is_paused) && !Boolean(data.pause_confirmed));
        if (data.current_progress) {
          setTrainingProgress(data.current_progress);
        }
      } else {
        isTrainingActiveRef.current = false;
        setIsTrainingActive(false);
        setIsTrainingPaused(false);
        setIsTrainingPausing(false);
        // If training is not active but we have completed results, process them
        // This handles the case where training completed while client was disconnected
        if (data.completed_results) {
          // Create a simple hash to avoid reprocessing the same results
          const resultsHash = JSON.stringify(data.completed_results.final_results?.training);
          if (processedCompletedResultsRef.current !== resultsHash) {
            console.log("Found completed training results from status check");
            processedCompletedResultsRef.current = resultsHash;
            setTrainingCompleted(data.completed_results);
            setIsTrainingActive(false);
          }
        }
      }
    });

    // Handle training pause/resume events
    newSocket.on("training_pausing", (data: any) => {
      console.log("Training pausing:", data);
      setIsTrainingPausing(true);
    });

    newSocket.on("training_paused", (data: any) => {
      console.log("Training paused:", data);
      setIsTrainingPausing(false);
      setIsTrainingPaused(true);
    });

    newSocket.on("training_resuming", (data: any) => {
      console.log("Training resuming:", data);
      setIsTrainingPausing(false);
    });

    newSocket.on("training_resumed", (data: any) => {
      console.log("Training resumed:", data);
      setIsTrainingPausing(false);
      setIsTrainingPaused(false);
    });

    newSocket.on("training_stopped", (data: any) => {
      console.log("Training stopped:", data);
      clearStoredJobId();
      isTrainingActiveRef.current = false;
      setIsTrainingPausing(false);
      setIsTrainingActive(false);
      setIsTrainingPaused(false);
    });

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("beforeunload", handleBeforeUnload);
      window.removeEventListener("pagehide", handlePageHide);
      newSocket.close();
    };
  }, []);

  const startTraining = async (config: any) => {
    const socket = socketRef.current;
    if (!socket?.connected || !socket.id) {
      setTrainingError("Not connected to training server");
      return;
    }

    try {
      const response = await fetch(API_CONFIG.getApiUrl("/train-stream"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Required when using ngrok endpoints to suppress browser interstitial
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ ...config, socket_id: socket.id }),
      });

      if (!response.ok) {
        let message = `Training request failed: ${response.status}`;
        try {
          const err = await response.json();
          if (typeof err.detail === "string") {
            message = err.detail;
          }
        } catch {
          // keep fallback message
        }
        throw new Error(message);
      }

      const result = await response.json();
      if (result.job_id) {
        writeStoredJobId(result.job_id);
      }
      console.log("Training started:", result);
    } catch (error) {
      console.error("Failed to start training:", error);
      setTrainingError(
        error instanceof Error ? error.message : "Failed to start training",
      );
    }
  };

  const pauseTraining = () => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("pause_training");
    }
  };

  const resumeTraining = () => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("resume_training");
    }
  };

  const stopTraining = () => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("stop_training");
    }
  };

  const resetTraining = () => {
    clearStoredJobId();
    setTrainingProgress(null);
    setTrainingCompleted(null);
    setTrainingError(null);
    setIsTrainingActive(false);
    setIsTrainingPaused(false);
    setIsTrainingPausing(false);
    processedCompletedResultsRef.current = null;
  };

  const checkTrainingStatus = () => {
    if (socketRef.current?.connected) {
      const jobId = readStoredJobId();
      if (jobId) {
        socketRef.current.emit("join_job", { job_id: jobId });
      } else {
        socketRef.current.emit("check_training_status");
      }
    }
  };

  return {
    socket,
    isConnected,
    trainingProgress,
    isTrainingActive,
    isTrainingPaused,
    isTrainingPausing,
    trainingCompleted,
    trainingError,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    resetTraining,
    checkTrainingStatus,
  };
};
