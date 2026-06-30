import { createFileRoute } from "@tanstack/react-router";
import { Link } from "@tanstack/react-router";
import { ArrowRight, Radar, Waves, ShieldCheck, Cpu } from "lucide-react";
import { UnderwaterBackground } from "@/components/trident/UnderwaterBackground";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TRIDENT — Underwater Acoustic Intelligence Platform" },
      { name: "description", content: "AI-powered underwater acoustic target classification and explainable threat intelligence." },
      { property: "og:title", content: "TRIDENT — Underwater Acoustic Intelligence" },
      { property: "og:description", content: "AI-powered underwater acoustic target classification and explainable threat intelligence." },
    ],
  }),
  component: Landing,
});

function Landing() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <UnderwaterBackground intensity="full" />

      <nav className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2.5">
          <div className="relative grid h-9 w-9 place-items-center rounded-md bg-gradient-to-br from-cyan/30 to-teal/10 glow-border">
            <Radar className="h-5 w-5 text-cyan" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-[0.28em]">TRIDENT</div>
            <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">Naval Intelligence</div>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-7 text-xs text-muted-foreground">
          <Link to="/about" className="hover:text-cyan transition">About</Link>
          <Link to="/model-statistics" className="hover:text-cyan transition">Model</Link>
          <a href="#capabilities" className="hover:text-cyan transition">Capabilities</a>
        </div>
        <Link
          to="/dashboard"
          className="group inline-flex items-center gap-2 rounded-md border border-cyan/40 bg-cyan/10 px-4 py-2 text-xs font-medium tracking-wider uppercase text-cyan transition hover:bg-cyan/20 glow-border"
        >
          Launch Console <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" />
        </Link>
      </nav>

      <section className="relative z-10 mx-auto flex max-w-6xl flex-col items-center px-6 pt-20 pb-32 text-center">
        <div className="mono inline-flex items-center gap-2 rounded-full border border-border bg-background/60 px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-muted-foreground backdrop-blur">
          <span className="size-1.5 rounded-full bg-threat-low animate-pulse" />
          Classified · Operational
        </div>

        <h1 className="mt-8 text-[clamp(3.5rem,11vw,9rem)] font-semibold leading-[0.95] tracking-[-0.04em] text-glow">
          TRIDENT
        </h1>
        <p className="mt-4 mono text-xs uppercase tracking-[0.5em] text-cyan/80">
          Underwater Acoustic Intelligence Platform
        </p>
        <p className="mt-8 max-w-2xl text-base text-muted-foreground sm:text-lg">
          AI-powered underwater acoustic target classification and explainable threat intelligence
          for naval surveillance, fleet defense, and maritime domain awareness.
        </p>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <Link
            to="/dashboard"
            className="group inline-flex items-center gap-2 rounded-md bg-cyan px-6 py-3 text-sm font-medium tracking-wide text-primary-foreground transition hover:bg-cyan/90 glow-border"
          >
            Enter Command Center <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
          </Link>
          <Link
            to="/about"
            className="inline-flex items-center gap-2 rounded-md border border-border bg-background/40 px-6 py-3 text-sm tracking-wide text-foreground/80 backdrop-blur hover:text-cyan transition"
          >
            Mission Brief
          </Link>
        </div>
      </section>

      <section id="capabilities" className="relative z-10 mx-auto grid max-w-6xl gap-4 px-6 pb-24 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { icon: Waves, title: "Passive Sonar Ingest", desc: "Raw hydrophone WAV streams analyzed via Mel-spectrogram pipelines." },
          { icon: Cpu, title: "CNN Inference", desc: "Sub-100ms classification of submarines, vessels, marine fauna, and ambient noise." },
          { icon: ShieldCheck, title: "Threat Doctrine", desc: "Doctrine-aligned threat scoring with recommended operational actions." },
          { icon: Radar, title: "Grad-CAM XAI", desc: "Per-prediction attention overlays for explainable analyst trust." },
        ].map(({ icon: Icon, title, desc }) => (
          <div key={title} className="glass-panel rounded-md p-5 transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-glow)]">
            <Icon className="h-5 w-5 text-cyan" strokeWidth={1.6} />
            <div className="mt-4 mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{title}</div>
            <p className="mt-2 text-sm leading-relaxed text-foreground/80">{desc}</p>
          </div>
        ))}
      </section>

      <div className="relative z-10 border-t border-border/50 bg-background/40 px-6 py-5 text-center backdrop-blur">
        <div className="mono text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
          TRIDENT v1.0.0 · Built with TensorFlow · MIT License
        </div>
      </div>
    </div>
  );
}
