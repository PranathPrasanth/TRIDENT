import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Panel, SectionTitle } from "@/components/trident/primitives";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_console/settings")({
  head: () => ({ meta: [{ title: "Settings — TRIDENT" }] }),
  component: Settings,
});

function Settings() {
  const [theme, setTheme] = useState<"dark" | "midnight">("dark");
  const [anim, setAnim] = useState(true);
  const [notif, setNotif] = useState(true);
  const [compact, setCompact] = useState(false);

  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Configuration" title="Settings" description="Operator preferences and console behavior." />

      <Panel title="Appearance">
        <div className="grid gap-3 sm:grid-cols-2">
          {(["dark", "midnight"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className={cn(
                "rounded-md border p-4 text-left transition",
                theme === t ? "border-cyan/60 bg-cyan/5 glow-border" : "border-border bg-background/40 hover:border-cyan/30",
              )}
            >
              <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Theme</div>
              <div className="mt-1 text-sm font-medium capitalize">{t === "dark" ? "Standard Dark" : "Midnight Operations"}</div>
            </button>
          ))}
        </div>
      </Panel>

      <Panel title="Behavior">
        <div className="space-y-2">
          <Toggle label="Animations" desc="Sonar pulses, motion transitions and counters." value={anim} onChange={setAnim} />
          <Toggle label="Notifications" desc="Alerts when a new high-confidence threat is detected." value={notif} onChange={setNotif} />
          <Toggle label="Compact Mode" desc="Tighter spacing for dense intelligence layouts." value={compact} onChange={setCompact} />
        </div>
      </Panel>

      <Panel title="API Endpoint">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Backend URL" value="https://api.trident.naval/v1" />
          <Field label="Inference Region" value="us-east-naval-2" />
        </div>
      </Panel>
    </div>
  );
}

function Toggle({ label, desc, value, onChange }: { label: string; desc: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="flex w-full items-center justify-between rounded border border-border/60 bg-background/40 p-4 text-left transition hover:border-cyan/30"
    >
      <div>
        <div className="text-sm font-medium text-foreground">{label}</div>
        <div className="mt-0.5 text-xs text-muted-foreground">{desc}</div>
      </div>
      <span className={cn("relative h-6 w-11 rounded-full transition", value ? "bg-cyan/80 glow-border" : "bg-secondary")}>
        <span className={cn("absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-background transition", value && "translate-x-5")} />
      </span>
    </button>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-3">
      <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <input defaultValue={value} className="mt-1 w-full bg-transparent mono text-sm text-foreground outline-none" />
    </div>
  );
}