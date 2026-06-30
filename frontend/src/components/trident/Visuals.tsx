import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Synthetic waveform — deterministic, no real audio. */
export function Waveform({
  className,
  height = 160,
  seed = 7,
}: {
  className?: string;
  height?: number;
  seed?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      // grid
      ctx.strokeStyle = "rgba(140,180,210,0.08)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 32) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // center line
      ctx.strokeStyle = "rgba(140,180,210,0.18)";
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

      // waveform
      const points = 1200;
      ctx.lineWidth = 1.4;
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "rgba(94,234,212,0.95)");
      grad.addColorStop(1, "rgba(56,189,248,0.95)");
      ctx.strokeStyle = grad;
      ctx.shadowBlur = 8;
      ctx.shadowColor = "rgba(94,234,212,0.55)";
      ctx.beginPath();
      for (let i = 0; i < points; i++) {
        const t = i / points;
        const x = t * w;
        const f1 = Math.sin((t * 80 + seed) * zoom) * Math.exp(-Math.pow(t - 0.3, 2) * 6);
        const f2 = Math.sin((t * 140 + seed * 2) * zoom) * 0.4 * Math.exp(-Math.pow(t - 0.65, 2) * 4);
        const noise = (Math.sin(i * 13.37 + seed) * 0.5 + Math.sin(i * 7.7) * 0.5) * 0.15;
        const y = h / 2 + ((f1 + f2 + noise) * h) / 2.4;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [zoom, seed]);

  return (
    <div className={cn("relative w-full", className)} style={{ height }}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <div className="absolute top-2 right-2 flex items-center gap-1 rounded border border-border bg-background/60 px-2 py-1 mono text-[10px] text-muted-foreground backdrop-blur">
        <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} className="px-1 hover:text-cyan">−</button>
        <span>{zoom.toFixed(2)}x</span>
        <button onClick={() => setZoom((z) => Math.min(3, z + 0.25))} className="px-1 hover:text-cyan">+</button>
      </div>
      <div className="absolute bottom-1 left-2 mono text-[9px] text-muted-foreground">0.00s — 4.20s · 22.05 kHz</div>
    </div>
  );
}

/** Synthetic Mel spectrogram. */
export function Spectrogram({
  className,
  height = 220,
  overlay = 0,
  showHeatmap = false,
  seed = 11,
}: {
  className?: string;
  height?: number;
  overlay?: number;
  showHeatmap?: boolean;
  seed?: number;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const heatRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const draw = (canvas: HTMLCanvasElement, mode: "spec" | "heat") => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);
      const cols = 180, rows = 80;
      const cw = w / cols, ch = h / rows;
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const tx = i / cols, ty = j / rows;
          let v =
            Math.sin((tx * 18 + seed) + ty * 4) * 0.5 +
            Math.sin((tx * 6 + seed * 1.7) * (ty * 3 + 1)) * 0.4 +
            Math.exp(-Math.pow(ty - 0.35, 2) * 12) * 0.7 * Math.sin(tx * 30 + seed) +
            Math.exp(-Math.pow(tx - 0.5, 2) * 5) * 0.6;
          v = (v + 1) / 2;
          v = Math.max(0, Math.min(1, v));
          if (mode === "spec") {
            // viridis-like
            const r = Math.round(20 + v * 40);
            const g = Math.round(40 + v * 200);
            const b = Math.round(120 + v * 130);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
          } else {
            const focus = Math.exp(-Math.pow(tx - 0.55, 2) * 8) * Math.exp(-Math.pow(ty - 0.4, 2) * 10);
            const intensity = Math.max(0, Math.min(1, focus * 1.4));
            const r = Math.round(255 * intensity);
            const g = Math.round(80 + 60 * (1 - intensity));
            const b = Math.round(40 + 40 * (1 - intensity));
            ctx.fillStyle = `rgba(${r},${g},${b},${intensity})`;
          }
          ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
        }
      }
    };
    if (ref.current) draw(ref.current, "spec");
    if (heatRef.current) draw(heatRef.current, "heat");
    const ro = new ResizeObserver(() => {
      if (ref.current) draw(ref.current, "spec");
      if (heatRef.current) draw(heatRef.current, "heat");
    });
    if (ref.current) ro.observe(ref.current);
    if (heatRef.current) ro.observe(heatRef.current);
    return () => ro.disconnect();
  }, [seed, showHeatmap]);

  return (
    <div className={cn("relative w-full overflow-hidden rounded-md border border-border", className)} style={{ height }}>
      <canvas ref={ref} className="absolute inset-0 h-full w-full" />
      {showHeatmap && (
        <canvas
          ref={heatRef}
          className="absolute inset-0 h-full w-full mix-blend-screen transition-opacity"
          style={{ opacity: overlay }}
        />
      )}
      <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-border/60" />
      <div className="absolute top-2 left-2 mono text-[10px] uppercase tracking-[0.2em] text-foreground/80">
        Mel Spectrogram · 128 bins
      </div>
      <div className="absolute bottom-2 right-2 mono text-[9px] text-foreground/60">8 kHz</div>
      <div className="absolute bottom-2 left-2 mono text-[9px] text-foreground/60">0 Hz</div>
    </div>
  );
}

export function ConfidenceRing({ value, size = 180, label }: { value: number; size?: number; label?: string }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setV(value));
    return () => cancelAnimationFrame(id);
  }, [value]);
  const r = (size - 16) / 2;
  const c = 2 * Math.PI * r;
  const off = c - (v / 100) * c;
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id="ring-grad" x1="0" x2="1">
            <stop offset="0%" stopColor="oklch(0.82 0.15 200)" />
            <stop offset="100%" stopColor="oklch(0.7 0.12 185)" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="oklch(0.3 0.04 240 / 0.5)" strokeWidth={6} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="url(#ring-grad)"
          strokeWidth={6}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={c}
          strokeDashoffset={off}
          style={{ transition: "stroke-dashoffset 1.4s cubic-bezier(0.22,1,0.36,1)", filter: "drop-shadow(0 0 8px oklch(0.78 0.14 200 / 0.6))" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="mono text-3xl font-semibold text-foreground">{value.toFixed(2)}<span className="text-cyan">%</span></div>
          {label && <div className="mono mt-0.5 text-[10px] uppercase tracking-[0.25em] text-muted-foreground">{label}</div>}
        </div>
      </div>
    </div>
  );
}

export function SonarLoader({ messages }: { messages: string[] }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((x) => (x + 1) % messages.length), 1100);
    return () => clearInterval(id);
  }, [messages.length]);
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-8">
      <div className="relative grid place-items-center" style={{ width: 160, height: 160 }}>
        <div className="absolute inset-0 rounded-full border border-cyan/30" />
        <div className="absolute inset-4 rounded-full border border-cyan/20" />
        <div className="absolute inset-8 rounded-full border border-cyan/15" />
        <div className="absolute inset-0 rounded-full overflow-hidden">
          <div
            className="absolute inset-0 animate-sweep"
            style={{
              background:
                "conic-gradient(from 0deg, transparent 0deg, oklch(0.78 0.14 200 / 0.5) 30deg, transparent 60deg)",
            }}
          />
        </div>
        <div className="size-2 rounded-full bg-cyan animate-pulse shadow-[0_0_16px_var(--cyan)]" />
      </div>
      <div className="mono text-xs uppercase tracking-[0.25em] text-cyan/90">{messages[i]}</div>
    </div>
  );
}