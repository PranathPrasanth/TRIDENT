import { useMemo } from "react";

export function UnderwaterBackground({ intensity = "full" }: { intensity?: "full" | "subtle" }) {
  const bubbles = useMemo(
    () =>
      Array.from({ length: intensity === "full" ? 28 : 12 }).map((_, i) => ({
        id: i,
        left: Math.random() * 100,
        size: 2 + Math.random() * 6,
        delay: Math.random() * 12,
        duration: 14 + Math.random() * 18,
        opacity: 0.15 + Math.random() * 0.35,
      })),
    [intensity],
  );

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-40" />
      <div
        className="absolute inset-x-0 bottom-0 h-[60%]"
        style={{
          background:
            "radial-gradient(ellipse at 50% 100%, oklch(0.32 0.08 220 / 0.35), transparent 70%)",
        }}
      />
      {intensity === "full" && (
        <>
          <div className="absolute left-1/2 top-1/2 size-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan/30 animate-sonar" />
          <div
            className="absolute left-1/2 top-1/2 size-[420px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan/20 animate-sonar"
            style={{ animationDelay: "1.5s" }}
          />
        </>
      )}
      {bubbles.map((b) => (
        <span
          key={b.id}
          className="absolute bottom-0 rounded-full bg-cyan/60"
          style={{
            left: `${b.left}%`,
            width: b.size,
            height: b.size,
            opacity: b.opacity,
            animation: `float-bubble ${b.duration}s linear ${b.delay}s infinite`,
            filter: "blur(0.5px)",
            boxShadow: "0 0 8px oklch(0.82 0.15 200 / 0.6)",
          }}
        />
      ))}
      <svg
        className="absolute inset-x-0 bottom-0 w-[200%] animate-wave-drift opacity-20"
        viewBox="0 0 1440 120"
        preserveAspectRatio="none"
        style={{ height: 120 }}
      >
        <path
          d="M0,60 C240,100 480,20 720,60 C960,100 1200,20 1440,60 L1440,120 L0,120 Z"
          fill="oklch(0.78 0.14 200)"
        />
      </svg>
    </div>
  );
}