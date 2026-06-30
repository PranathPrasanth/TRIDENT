import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Panel, SectionTitle } from "@/components/trident/primitives";
import { Spectrogram } from "@/components/trident/Visuals";
import { Maximize2, Eye, EyeOff } from "lucide-react";

export const Route = createFileRoute("/_console/explainability")({
  head: () => ({ meta: [{ title: "Explainability — TRIDENT" }] }),
  component: Explainability,
});

function Explainability() {
  const [opacity, setOpacity] = useState(0.65);
  const [overlay, setOverlay] = useState(true);
  const [full, setFull] = useState(false);

  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Explainable AI" title="Grad-CAM Attention" description="Inspect which spectro-temporal regions drove the CNN's classification decision." />

      <Panel
        title="Controls"
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOverlay((o) => !o)}
              className="inline-flex items-center gap-1.5 rounded border border-border bg-background/40 px-3 py-1.5 mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-cyan transition"
            >
              {overlay ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              {overlay ? "Hide" : "Show"} Overlay
            </button>
            <button
              onClick={() => setFull((f) => !f)}
              className="inline-flex items-center gap-1.5 rounded border border-border bg-background/40 px-3 py-1.5 mono text-[11px] uppercase tracking-wider text-muted-foreground hover:text-cyan transition"
            >
              <Maximize2 className="h-3.5 w-3.5" /> Fullscreen
            </button>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-3 text-sm">
            <span className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground w-24">Opacity</span>
            <input
              type="range" min={0} max={1} step={0.01} value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-56 accent-cyan"
            />
            <span className="mono text-cyan w-12">{Math.round(opacity * 100)}%</span>
          </label>
        </div>
      </Panel>

      <div className={`grid gap-4 ${full ? "" : "lg:grid-cols-2"}`}>
        <Panel title="Original Mel Spectrogram">
          <Spectrogram height={full ? 480 : 320} />
        </Panel>
        <Panel title="Grad-CAM Overlay">
          <Spectrogram height={full ? 480 : 320} showHeatmap overlay={overlay ? opacity : 0} seed={11} />
        </Panel>
      </div>

      <Panel title="Attention Summary">
        <div className="grid gap-4 md:grid-cols-3 text-sm">
          <Insight label="Peak Activation Band" value="380 Hz – 1.2 kHz" />
          <Insight label="Dominant Time Window" value="1.8 s – 2.6 s" />
          <Insight label="Class Contribution" value="SUBMARINE · 0.91" />
        </div>
        <p className="mt-4 text-sm text-foreground/80">
          The model focused on a sustained low-frequency tonal in the 380 Hz – 1.2 kHz band consistent
          with a diesel-electric submarine cavitation signature. Secondary harmonic structure between
          1.8 s and 2.6 s reinforced the classification.
        </p>
      </Panel>
    </div>
  );
}

function Insight({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-4">
      <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <div className="mt-1 mono text-base text-foreground">{value}</div>
    </div>
  );
}