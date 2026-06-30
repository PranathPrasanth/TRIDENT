import { createFileRoute } from "@tanstack/react-router";
import { Panel, SectionTitle } from "@/components/trident/primitives";
import { Waveform, Spectrogram } from "@/components/trident/Visuals";

export const Route = createFileRoute("/_console/acoustic-analysis")({
  head: () => ({ meta: [{ title: "Acoustic Analysis — TRIDENT" }] }),
  component: AcousticAnalysis,
});

function AcousticAnalysis() {
  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Signal Domain" title="Acoustic Analysis" description="Time-domain waveform inspection and Mel-spectrogram decomposition of captured hydrophone signals." />

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Capture · Hydrophone 07-A" className="lg:col-span-2">
          <Waveform height={200} />
        </Panel>
        <Panel title="Signal Metrics">
          <dl className="grid grid-cols-2 gap-3 text-sm">
            {[
              ["Duration", "4.20 s"],
              ["Sample Rate", "22.05 kHz"],
              ["Channels", "Mono"],
              ["Bit Depth", "16-bit"],
              ["RMS", "−18.4 dB"],
              ["Peak", "−2.1 dB"],
              ["SNR", "27.8 dB"],
              ["Centroid", "1.84 kHz"],
            ].map(([k, v]) => (
              <div key={k} className="rounded border border-border/60 bg-background/40 p-3">
                <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{k}</div>
                <div className="mono mt-1 text-base text-foreground">{v}</div>
              </div>
            ))}
          </dl>
        </Panel>
      </div>

      <Panel title="Mel Spectrogram · 128 bins">
        <Spectrogram height={300} />
      </Panel>

      <div className="grid gap-4 md:grid-cols-3">
        <Panel title="Low Band · 0–500 Hz"><Spectrogram height={140} seed={3} /></Panel>
        <Panel title="Mid Band · 500–2k Hz"><Spectrogram height={140} seed={17} /></Panel>
        <Panel title="High Band · 2k–8k Hz"><Spectrogram height={140} seed={29} /></Panel>
      </div>
    </div>
  );
}