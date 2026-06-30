import { useEffect, useRef, useState, type ReactNode, type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Panel({
  title,
  subtitle,
  action,
  children,
  className,
  ...rest
}: HTMLAttributes<HTMLDivElement> & { title?: ReactNode; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <section
      {...rest}
      className={cn("glass-panel rounded-lg p-5 transition-shadow hover:shadow-[var(--shadow-glow)]", className)}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title && (
              <h2 className="mono text-[11px] uppercase tracking-[0.28em] text-muted-foreground">{title}</h2>
            )}
            {subtitle && <div className="mt-1 text-sm text-foreground/80">{subtitle}</div>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function AnimatedCounter({
  value,
  decimals = 0,
  suffix = "",
  duration = 1200,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
}) {
  const [n, setN] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setN(value * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return (
    <span className="mono tabular-nums">
      {n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      {suffix}
    </span>
  );
}

export type ThreatLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

const THREAT_STYLES: Record<ThreatLevel, { color: string; bg: string; ring: string; label: string }> = {
  LOW: { color: "text-threat-low", bg: "bg-threat-low/10", ring: "ring-threat-low/40", label: "Minimal Risk" },
  MEDIUM: { color: "text-threat-medium", bg: "bg-threat-medium/10", ring: "ring-threat-medium/40", label: "Elevated" },
  HIGH: { color: "text-threat-high", bg: "bg-threat-high/10", ring: "ring-threat-high/40", label: "Hostile" },
  CRITICAL: { color: "text-threat-critical", bg: "bg-threat-critical/15", ring: "ring-threat-critical/50", label: "Imminent" },
};

export function ThreatBadge({ level, size = "md" }: { level: ThreatLevel; size?: "sm" | "md" | "lg" }) {
  const s = THREAT_STYLES[level];
  const sz = size === "lg" ? "text-base px-4 py-2" : size === "sm" ? "text-[10px] px-2 py-0.5" : "text-xs px-2.5 py-1";
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-sm uppercase tracking-[0.22em] mono ring-1", s.color, s.bg, s.ring, sz)}>
      <span className={cn("size-1.5 rounded-full animate-pulse", s.color.replace("text-", "bg-"))} />
      {level}
    </span>
  );
}

export function threatStyles(level: ThreatLevel) {
  return THREAT_STYLES[level];
}

export function SectionTitle({ eyebrow, title, description }: { eyebrow?: string; title: string; description?: string }) {
  return (
    <div className="mb-6">
      {eyebrow && <div className="mono text-[10px] uppercase tracking-[0.3em] text-cyan/80">{eyebrow}</div>}
      <h1 className="mt-1 text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">{title}</h1>
      {description && <p className="mt-2 max-w-2xl text-sm text-muted-foreground">{description}</p>}
    </div>
  );
}

export function useInView<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    const obs = new IntersectionObserver(([e]) => e.isIntersecting && setSeen(true), { threshold: 0.2 });
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return { ref, seen };
}