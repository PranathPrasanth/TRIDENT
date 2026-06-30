import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Activity, Database, Layers, Target, Gauge, ShieldAlert, FileAudio2 } from "lucide-react";
import { Panel, AnimatedCounter, ThreatBadge, SectionTitle, threatStyles, type ThreatLevel } from "@/components/trident/primitives";
import { Waveform, Spectrogram, ConfidenceRing, SonarLoader } from "@/components/trident/Visuals";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_console/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard — TRIDENT" }] }),
  component: Dashboard,
});

type FileMeta = { name: string; size: number; duration: number; rate: number };

function Dashboard() {
  const [file, setFile] = useState<FileMeta | null>(null);
  const [progress, setProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<{ target: string; confidence: number; inference: number; threat: ThreatLevel } | null>(null);

  const runAnalysis = () => {
    setAnalyzing(true);
    setTimeout(() => {
      setAnalyzing(false);
      setResult({ target: "SUBMARINE", confidence: 98.74, inference: 84, threat: "HIGH" });
    }, 3600);
  };

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setResult(null);
    setProgress(0);
    const meta: FileMeta = { name: f.name, size: f.size, duration: 4.2, rate: 22050 };
    const id = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) { clearInterval(id); setFile(meta); runAnalysis(); return 100; }
        return p + 8;
      });
    }, 60);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "audio/wav": [".wav"] }, multiple: false,
  });

  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Command Center" title="Operational Overview" description="Real-time underwater acoustic intelligence, classification, and threat posture." />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard icon={Activity} label="Model Status" value="ONLINE" tone="cyan" pulse />
        <StatCard icon={Gauge} label="Model Accuracy" value={<><AnimatedCounter value={97.82} decimals={2} />%</>} />
        <StatCard icon={Layers} label="Dataset Classes" value={<AnimatedCounter value={12} />} />
        <StatCard icon={Database} label="Audio Samples" value={<AnimatedCounter value={14820} />} />
        <StatCard icon={Target} label="Latest Prediction" value={result?.target ?? "—"} mono />
        <StatCard icon={ShieldAlert} label="Threat Level" value={<ThreatBadge level={result?.threat ?? "LOW"} />} />
      </div>

      <Panel
        title="Acoustic Capture · Hydrophone Ingest"
        subtitle="Drop a passive sonar recording to begin classification"
      >
        <div
          {...getRootProps()}
          className={cn(
            "relative grid place-items-center rounded-md border border-dashed border-border bg-background/30 px-6 py-10 text-center transition",
            isDragActive && "border-cyan bg-cyan/5 glow-border",
          )}
        >
          <input {...getInputProps()} />
          <div className="grid h-14 w-14 place-items-center rounded-full bg-cyan/10 ring-1 ring-cyan/30">
            <UploadCloud className="h-6 w-6 text-cyan" />
          </div>
          <div className="mt-4 text-sm">
            <span className="text-foreground">Drag &amp; drop </span>
            <span className="text-muted-foreground">a .wav file, or </span>
            <span className="text-cyan">browse</span>
          </div>
          <div className="mono mt-1 text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
            Accepted · WAV · 16-bit PCM · ≤ 30s
          </div>
          {progress > 0 && progress < 100 && (
            <div className="mt-5 h-1 w-64 overflow-hidden rounded bg-secondary">
              <div className="h-full bg-gradient-to-r from-cyan to-teal transition-all" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>

        {file && (
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            <FileFact label="Filename" value={file.name} icon={FileAudio2} />
            <FileFact label="Duration" value={`${file.duration.toFixed(2)} s`} />
            <FileFact label="Sample Rate" value={`${(file.rate / 1000).toFixed(2)} kHz`} />
            <FileFact label="File Size" value={`${(file.size / 1024).toFixed(1)} KB`} />
          </div>
        )}
      </Panel>

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

      {result && !analyzing && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Panel title="Acoustic Signal · Waveform" className="lg:col-span-2">
            <Waveform />
          </Panel>
          <Panel title="AI Prediction">
            <div className="flex flex-col items-center gap-4">
              <ConfidenceRing value={result.confidence} label="Confidence" />
              <div className="text-center">
                <div className="mono text-[10px] uppercase tracking-[0.3em] text-muted-foreground">Predicted Target</div>
                <div className="mt-1 text-2xl font-semibold tracking-wide text-foreground">{result.target}</div>
                <div className="mono mt-2 text-[11px] text-muted-foreground">
                  Inference · <span className="text-cyan">{result.inference} ms</span>
                </div>
              </div>
            </div>
          </Panel>
          <Panel title="Mel Spectrogram" className="lg:col-span-2">
            <Spectrogram />
          </Panel>
          <Panel title="Threat Intelligence">
            <ThreatCard target={result.target} level={result.threat} />
          </Panel>
        </div>
      )}

      <Panel title="Recent Analyses · Timeline">
        <TimelineTable />
      </Panel>
    </div>
  );
}

function StatCard({
  icon: Icon, label, value, tone, pulse, mono,
}: { icon: any; label: string; value: React.ReactNode; tone?: "cyan"; pulse?: boolean; mono?: boolean }) {
  return (
    <div className="glass-panel relative overflow-hidden rounded-lg p-4 transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-glow)]">
      <div className="flex items-center justify-between">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
        <Icon className={cn("h-4 w-4", tone === "cyan" ? "text-cyan" : "text-muted-foreground")} strokeWidth={1.6} />
      </div>
      <div className={cn("mt-3 text-2xl font-semibold tracking-tight text-foreground", mono && "mono text-xl")}>
        {value}
      </div>
      {pulse && (
        <div className="absolute -bottom-px left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan/60 to-transparent" />
      )}
    </div>
  );
}

function FileFact({ label, value, icon: Icon }: { label: string; value: string; icon?: any }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-3">
      <div className="mono flex items-center gap-1.5 text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {Icon && <Icon className="h-3 w-3" />} {label}
      </div>
      <div className="mono mt-1 truncate text-sm text-foreground">{value}</div>
    </div>
  );
}

function ThreatCard({ target, level }: { target: string; level: ThreatLevel }) {
  const s = threatStyles(level);
  const actions: Record<ThreatLevel, { action: string; note: string }> = {
    LOW: { action: "Continue passive monitoring", note: "Signature consistent with ambient or civilian traffic." },
    MEDIUM: { action: "Escalate to watch officer", note: "Persistent contact — recommend extended surveillance window." },
    HIGH: { action: "Alert tactical command", note: "Hostile signature profile. Maintain contact, prepare countermeasures." },
    CRITICAL: { action: "Immediate defensive posture", note: "Imminent threat. Engage rules of engagement protocol." },
  };
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Category</div>
          <div className="mt-0.5 text-base font-semibold text-foreground">{target}</div>
        </div>
        <ThreatBadge level={level} size="lg" />
      </div>
      <div className={cn("rounded border p-3 ring-1 border-transparent", s.bg, s.ring)}>
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Recommended Action</div>
        <div className="mt-1 text-sm text-foreground">{actions[level].action}</div>
      </div>
      <div className="rounded border border-border/60 bg-background/40 p-3">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Operational Notes</div>
        <div className="mt-1 text-sm text-foreground/85">{actions[level].note}</div>
      </div>
    </div>
  );
}

const TIMELINE: Array<{ time: string; pred: string; conf: number; level: ThreatLevel }> = [
  { time: "14:08:42", pred: "SUBMARINE", conf: 98.74, level: "HIGH" },
  { time: "13:51:11", pred: "CARGO_VESSEL", conf: 91.20, level: "LOW" },
  { time: "13:32:05", pred: "WHALE_CALL", conf: 87.55, level: "LOW" },
  { time: "13:10:47", pred: "TORPEDO", conf: 96.41, level: "CRITICAL" },
  { time: "12:48:22", pred: "FRIGATE", conf: 88.97, level: "MEDIUM" },
  { time: "12:21:09", pred: "AMBIENT_NOISE", conf: 94.10, level: "LOW" },
];

function TimelineTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="mono text-left text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            <th className="py-2 pr-4 font-medium">Timestamp</th>
            <th className="py-2 pr-4 font-medium">Prediction</th>
            <th className="py-2 pr-4 font-medium">Confidence</th>
            <th className="py-2 pr-4 font-medium">Threat</th>
          </tr>
        </thead>
        <tbody>
          {TIMELINE.map((r) => (
            <tr key={r.time} className="border-t border-border/40 hover:bg-cyan/5 transition">
              <td className="py-2.5 pr-4 mono text-muted-foreground">{r.time} UTC</td>
              <td className="py-2.5 pr-4 font-medium tracking-wide">{r.pred}</td>
              <td className="py-2.5 pr-4 mono text-cyan">{r.conf.toFixed(2)}%</td>
              <td className="py-2.5 pr-4"><ThreatBadge level={r.level} size="sm" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}