import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

import {
  UploadCloud,
  Activity,
  Database,
  Layers,
  Target,
  Gauge,
  ShieldAlert,
  FileAudio2,
} from "lucide-react";

import {
  Panel,
  ThreatBadge,
  SectionTitle,
  threatStyles,
  type ThreatLevel,
} from "@/components/trident/primitives";

import {
  Waveform,
  Spectrogram,
  ConfidenceRing,
  SonarLoader,
} from "@/components/trident/Visuals";

import { cn } from "@/lib/utils";

import {
  predictAudio,
  type PredictionResponse,
} from "@/lib/api";

export const Route = createFileRoute(
  "/_console/dashboard",
)({
  head: () => ({
    meta: [{ title: "Dashboard — TRIDENT" }],
  }),
  component: Dashboard,
});

type FileMeta = {
  name: string;
  size: number;
  duration: number;
  rate: number;
};

function Dashboard() {
  const [file, setFile] =
    useState<FileMeta | null>(null);

  const [progress, setProgress] = useState(0);

  const [analyzing, setAnalyzing] =
    useState(false);

  const [result, setResult] =
    useState<PredictionResponse | null>(null);

  const [error, setError] =
    useState<string | null>(null);

  const runAnalysis = async (
    audioFile: File,
  ) => {
    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response =
        await predictAudio(audioFile);

      setResult(response);

      setFile({
        name: response.file.name,
        size: response.file.size,
        duration: response.file.duration,
        rate: response.file.sample_rate,
      });

      setProgress(100);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to analyze audio.";

      setError(message);
    } finally {
      setAnalyzing(false);
    }
  };

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const selectedFile =
        acceptedFiles[0];

      if (!selectedFile) {
        return;
      }

      setResult(null);
      setError(null);
      setProgress(100);

      setFile({
        name: selectedFile.name,
        size: selectedFile.size,
        duration: 0,
        rate: 0,
      });

      void runAnalysis(selectedFile);
    },
    [],
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
  } = useDropzone({
    onDrop,
    accept: {
      "audio/wav": [".wav"],
    },
    multiple: false,
  });

  /*
   * The backend currently returns UNKNOWN for
   * threat profiles that do not have a matching
   * profile. The existing UI ThreatBadge only
   * supports the four defined ThreatLevel values,
   * so UNKNOWN is displayed using the neutral LOW
   * visual style while the actual backend value
   * remains visible in the ThreatCard.
   */
  const uiThreatLevel: ThreatLevel =
    result?.threat.threat_level === "MEDIUM"
      ? "MEDIUM"
      : result?.threat.threat_level === "HIGH"
        ? "HIGH"
        : result?.threat.threat_level === "CRITICAL"
          ? "CRITICAL"
          : "LOW";

  return (
    <div className="space-y-6">
      <SectionTitle
        eyebrow="Command Center"
        title="Operational Overview"
        description="Real-time underwater acoustic intelligence, classification, and threat posture."
      />

      {/* =====================================================
          SYSTEM STATUS
      ====================================================== */}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard
          icon={Activity}
          label="Model Status"
          value="ONLINE"
          tone="cyan"
          pulse
        />

        <StatCard
          icon={Gauge}
          label="Model Accuracy"
          value="44.44%"
        />

        <StatCard
          icon={Layers}
          label="Dataset Classes"
          value="4"
        />

        <StatCard
          icon={Database}
          label="Audio Samples"
          value="63"
        />

        <StatCard
          icon={Target}
          label="Latest Prediction"
          value={
            result
              ? result.prediction.target.toUpperCase()
              : "—"
          }
          mono
        />

        <StatCard
          icon={ShieldAlert}
          label="Threat Level"
          value={
            <ThreatBadge
              level={uiThreatLevel}
            />
          }
        />
      </div>

      {/* =====================================================
          AUDIO UPLOAD
      ====================================================== */}

      <Panel
        title="Acoustic Capture · Hydrophone Ingest"
        subtitle="Drop a passive sonar recording to begin classification"
      >
        <div
          {...getRootProps()}
          className={cn(
            "relative grid place-items-center rounded-md border border-dashed border-border bg-background/30 px-6 py-10 text-center transition",
            isDragActive &&
              "border-cyan bg-cyan/5 glow-border",
          )}
        >
          <input {...getInputProps()} />

          <div className="grid h-14 w-14 place-items-center rounded-full bg-cyan/10 ring-1 ring-cyan/30">
            <UploadCloud className="h-6 w-6 text-cyan" />
          </div>

          <div className="mt-4 text-sm">
            <span className="text-foreground">
              Drag &amp; drop{" "}
            </span>

            <span className="text-muted-foreground">
              a .wav file, or{" "}
            </span>

            <span className="text-cyan">
              browse
            </span>
          </div>

          <div className="mono mt-1 text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
            Accepted · WAV · 16-bit PCM · ≤ 30s
          </div>

          {progress > 0 &&
            progress < 100 && (
              <div className="mt-5 h-1 w-64 overflow-hidden rounded bg-secondary">
                <div
                  className="h-full bg-gradient-to-r from-cyan to-teal transition-all"
                  style={{
                    width: `${progress}%`,
                  }}
                />
              </div>
            )}
        </div>

        {file && (
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            <FileFact
              label="Filename"
              value={file.name}
              icon={FileAudio2}
            />

            <FileFact
              label="Duration"
              value={
                file.duration > 0
                  ? `${file.duration.toFixed(2)} s`
                  : "Processing..."
              }
            />

            <FileFact
              label="Sample Rate"
              value={
                file.rate > 0
                  ? `${(
                      file.rate / 1000
                    ).toFixed(2)} kHz`
                  : "Processing..."
              }
            />

            <FileFact
              label="File Size"
              value={`${(
                file.size / 1024
              ).toFixed(1)} KB`}
            />
          </div>
        )}
      </Panel>

      {/* =====================================================
          ERROR
      ====================================================== */}

      {error && (
        <Panel title="Analysis Error">
          <div className="rounded border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            {error}
          </div>
        </Panel>
      )}

      {/* =====================================================
          REAL INFERENCE LOADER
      ====================================================== */}

      {analyzing && (
        <Panel title="Inference Pipeline">
          <SonarLoader
            messages={[
              "Analyzing underwater acoustic signature…",
              "Extracting Mel Spectrogram…",
              "Running CNN inference…",
              "Generating Explainability…",
              "Threat Assessment Complete.",
            ]}
          />
        </Panel>
      )}

      {/* =====================================================
          REAL RESULT
      ====================================================== */}

      {result && !analyzing && (
        <div className="grid gap-4 lg:grid-cols-3">

          {/* Waveform */}

          <Panel
            title="Acoustic Signal · Waveform"
            className="lg:col-span-2"
          >
            <Waveform />
          </Panel>

          {/* Prediction */}

          <Panel title="AI Prediction">
            <div className="flex flex-col items-center gap-4">

              <ConfidenceRing
                value={
                  result.prediction
                    .confidence_percent
                }
                label="Confidence"
              />

              <div className="text-center">

                <div className="mono text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
                  Predicted Target
                </div>

                <div className="mt-1 text-2xl font-semibold tracking-wide text-foreground">
                  {result.prediction.target.toUpperCase()}
                </div>

                <div className="mono mt-2 text-[11px] text-muted-foreground">
                  Inference ·{" "}
                  <span className="text-cyan">
                    {result.inference.milliseconds.toFixed(
                      0,
                    )}{" "}
                    ms
                  </span>
                </div>

              </div>
            </div>
          </Panel>

          {/* Spectrogram */}

          <Panel
            title="Mel Spectrogram"
            className="lg:col-span-2"
          >
            <Spectrogram />
          </Panel>

          {/* Threat */}

          <Panel title="Threat Intelligence">
            <ThreatCard
              target={
                result.prediction.target
              }
              level={uiThreatLevel}
              backendThreatLevel={
                result.threat.threat_level
              }
              recommendedAction={
                result.threat
                  .recommended_action
              }
            />
          </Panel>

          {/* Probability distribution */}

          <Panel
            title="Class Probability Distribution"
            className="lg:col-span-3"
          >
            <ProbabilityTable
              probabilities={
                result.prediction
                  .probabilities
              }
            />
          </Panel>

          {/* Grad-CAM */}

          {result.explainability
            .gradcam_available &&
            result.explainability
              .heatmap && (
              <Panel
                title="Grad-CAM Explainability"
                className="lg:col-span-3"
              >
                <div className="flex flex-col gap-3">
                  <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    CNN attention heatmap ·{" "}
                    {result.prediction.target}
                  </div>

                  <img
                    src={
                      result.explainability
                        .heatmap
                    }
                    alt="Grad-CAM heatmap showing CNN attention regions"
                    className="mx-auto max-h-[420px] w-full rounded-md border border-border/60 object-contain"
                  />
                </div>
              </Panel>
            )}
        </div>
      )}

      {/* =====================================================
          TIMELINE
      ====================================================== */}

      <Panel title="Recent Analyses · Timeline">
        <TimelineTable />
      </Panel>
    </div>
  );
}

/* ============================================================
   STAT CARD
============================================================ */

function StatCard({
  icon: Icon,
  label,
  value,
  tone,
  pulse,
  mono,
}: {
  icon: any;
  label: string;
  value: React.ReactNode;
  tone?: "cyan";
  pulse?: boolean;
  mono?: boolean;
}) {
  return (
    <div className="glass-panel relative overflow-hidden rounded-lg p-4 transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-glow)]">

      <div className="flex items-center justify-between">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          {label}
        </div>

        <Icon
          className={cn(
            "h-4 w-4",
            tone === "cyan"
              ? "text-cyan"
              : "text-muted-foreground",
          )}
          strokeWidth={1.6}
        />
      </div>

      <div
        className={cn(
          "mt-3 text-2xl font-semibold tracking-tight text-foreground",
          mono && "mono text-xl",
        )}
      >
        {value}
      </div>

      {pulse && (
        <div className="absolute -bottom-px left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan/60 to-transparent" />
      )}
    </div>
  );
}

/* ============================================================
   FILE FACT
============================================================ */

function FileFact({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string;
  icon?: any;
}) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-3">
      <div className="mono flex items-center gap-1.5 text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {Icon && (
          <Icon className="h-3 w-3" />
        )}
        {label}
      </div>

      <div className="mono mt-1 truncate text-sm text-foreground">
        {value}
      </div>
    </div>
  );
}

/* ============================================================
   THREAT CARD
============================================================ */

function ThreatCard({
  target,
  level,
  backendThreatLevel,
  recommendedAction,
}: {
  target: string;
  level: ThreatLevel;
  backendThreatLevel: string;
  recommendedAction: string;
}) {
  const s = threatStyles(level);

  const notes: Record<
    ThreatLevel,
    string
  > = {
    LOW:
      "Low-confidence or unclassified signature. Continue passive monitoring.",
    MEDIUM:
      "Persistent contact — recommend extended surveillance window.",
    HIGH:
      "Elevated signature classification. Maintain contact and review.",
    CRITICAL:
      "Critical classification. Follow established operational procedures.",
  };

  return (
    <div className="space-y-3">

      <div className="flex items-center justify-between">

        <div>
          <div className="mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
            Category
          </div>

          <div className="mt-0.5 text-base font-semibold text-foreground">
            {target.toUpperCase()}
          </div>
        </div>

        <ThreatBadge
          level={level}
          size="lg"
        />
      </div>

      <div
        className={cn(
          "rounded border p-3 ring-1 border-transparent",
          s.bg,
          s.ring,
        )}
      >
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Backend Threat Level
        </div>

        <div className="mt-1 text-sm text-foreground">
          {backendThreatLevel}
        </div>
      </div>

      <div className="rounded border border-border/60 bg-background/40 p-3">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Recommended Action
        </div>

        <div className="mt-1 text-sm text-foreground">
          {recommendedAction}
        </div>
      </div>

      <div className="rounded border border-border/60 bg-background/40 p-3">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Operational Notes
        </div>

        <div className="mt-1 text-sm text-foreground/85">
          {notes[level]}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   PROBABILITY TABLE
============================================================ */

function ProbabilityTable({
  probabilities,
}: {
  probabilities: Record<
    string,
    number
  >;
}) {
  const entries = Object.entries(
    probabilities,
  ).sort(
    ([, a], [, b]) => b - a,
  );

  return (
    <div className="space-y-3">
      {entries.map(
        ([target, probability]) => {
          const percentage =
            probability * 100;

          return (
            <div
              key={target}
              className="grid grid-cols-[140px_1fr_70px] items-center gap-3"
            >
              <div className="mono text-xs uppercase tracking-wide text-muted-foreground">
                {target}
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan to-teal transition-all"
                  style={{
                    width: `${percentage}%`,
                  }}
                />
              </div>

              <div className="mono text-right text-xs text-cyan">
                {percentage.toFixed(2)}%
              </div>
            </div>
          );
        },
      )}
    </div>
  );
}

/* ============================================================
   TIMELINE
============================================================ */

/*
 * These are still placeholder historical entries.
 * They are intentionally kept separate from the
 * real-time prediction result.
 *
 * We will replace this with backend history later.
 */

const TIMELINE: Array<{
  time: string;
  pred: string;
  conf: number;
  level: ThreatLevel;
}> = [
  {
    time: "14:08:42",
    pred: "DEMO CONTACT",
    conf: 65.94,
    level: "LOW",
  },
  {
    time: "13:51:11",
    pred: "CARGO_SHIP",
    conf: 61.20,
    level: "LOW",
  },
  {
    time: "13:32:05",
    pred: "TANKER",
    conf: 72.55,
    level: "LOW",
  },
];

/* ============================================================
   TIMELINE TABLE
============================================================ */

function TimelineTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="mono text-left text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            <th className="py-2 pr-4 font-medium">
              Timestamp
            </th>

            <th className="py-2 pr-4 font-medium">
              Prediction
            </th>

            <th className="py-2 pr-4 font-medium">
              Confidence
            </th>

            <th className="py-2 pr-4 font-medium">
              Threat
            </th>
          </tr>
        </thead>

        <tbody>
          {TIMELINE.map((row) => (
            <tr
              key={row.time}
              className="border-t border-border/40 transition hover:bg-cyan/5"
            >
              <td className="mono py-2.5 pr-4 text-muted-foreground">
                {row.time} UTC
              </td>

              <td className="py-2.5 pr-4 font-medium tracking-wide">
                {row.pred}
              </td>

              <td className="mono py-2.5 pr-4 text-cyan">
                {row.conf.toFixed(2)}%
              </td>

              <td className="py-2.5 pr-4">
                <ThreatBadge
                  level={row.level}
                  size="sm"
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}