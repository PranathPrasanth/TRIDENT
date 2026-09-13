const API_BASE_URL = "http://127.0.0.1:8000";

export type PredictionResponse = {
  success: boolean;

  file: {
    name: string;
    size: number;
    duration: number;
    sample_rate: number;
  };

  prediction: {
    target: string;
    confidence: number;
    confidence_percent: number;
    probabilities: Record<string, number>;
  };

  threat: {
    target: string;
    confidence: number;
    category: string;
    threat_level: string;
    recommended_action: string;
  };

  inference: {
    milliseconds: number;
    input_shape: number[];
  };

  explainability: {
    gradcam_available: boolean;
    heatmap: string | null;
  };
};

export async function predictAudio(
  file: File,
): Promise<PredictionResponse> {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/predict`,
    {
      method: "POST",
      body: formData,
    },
  );

  if (!response.ok) {
    let message = `API request failed (${response.status})`;

    try {
      const error = await response.json();

      if (error?.detail) {
        message = error.detail;
      }
    } catch {
      // Keep default error message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<PredictionResponse>;
}