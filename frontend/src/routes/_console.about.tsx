import { createFileRoute } from "@tanstack/react-router";
import { Panel, SectionTitle } from "@/components/trident/primitives";
import { Waves, Cpu, Eye, ShieldAlert, Radar } from "lucide-react";

export const Route = createFileRoute("/_console/about")({
  head: () => ({ meta: [{ title: "About — TRIDENT" }] }),
  component: About,
});

const PIPELINE = [
  { icon: Waves, title: "Acoustic Ingest", desc: "Hydrophone WAV streams · 22.05 kHz" },
  { icon: Radar, title: "Mel Spectrogram", desc: "128-bin time-frequency decomposition" },
  { icon: Cpu, title: "CNN Inference", desc: "TRIDENT-CNN classifier · 12 layers" },
  { icon: Eye, title: "Grad-CAM XAI", desc: "Per-prediction attention overlay" },
  { icon: ShieldAlert, title: "Threat Doctrine", desc: "Doctrine-aligned risk scoring" },
];

const STACK = ["TensorFlow", "Python", "Grad-CAM", "Streamlit / FastAPI", "Deep Learning", "Acoustic Signal Processing"];

function About() {
  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Mission" title="About TRIDENT" description="Naval-grade explainable AI for underwater acoustic surveillance." />

      <Panel title="Mission Statement">
        <p className="text-base leading-relaxed text-foreground/85">
          TRIDENT is an underwater acoustic intelligence platform engineered for naval analysts and
          maritime defense operators. It transforms raw passive sonar recordings into auditable
          classifications and doctrine-aligned threat assessments — combining deep learning with
          explainability so every decision can be inspected, justified, and trusted.
        </p>
      </Panel>

      <Panel title="Inference Pipeline">
        <ol className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {PIPELINE.map((p, i) => (
            <li key={p.title} className="relative rounded-md border border-border/60 bg-background/40 p-4">
              <div className="mono text-[10px] uppercase tracking-[0.25em] text-cyan">Stage {i + 1}</div>
              <p.icon className="mt-3 h-5 w-5 text-cyan" strokeWidth={1.6} />
              <div className="mt-3 text-sm font-medium">{p.title}</div>
              <div className="mono mt-1 text-[11px] text-muted-foreground">{p.desc}</div>
            </li>
          ))}
        </ol>
      </Panel>

      <Panel title="Technology Stack">
        <div className="flex flex-wrap gap-2">
          {STACK.map((s) => (
            <span key={s} className="rounded border border-border/60 bg-background/40 px-3 py-1.5 mono text-[11px] uppercase tracking-wider text-foreground/85">
              {s}
            </span>
          ))}
        </div>
      </Panel>
    </div>
  );
}