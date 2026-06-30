import { createFileRoute } from "@tanstack/react-router";
import { Panel, SectionTitle, ThreatBadge, threatStyles, type ThreatLevel } from "@/components/trident/primitives";
import { cn } from "@/lib/utils";
import { Crosshair, AlertTriangle, ShieldAlert, ShieldCheck } from "lucide-react";

export const Route = createFileRoute("/_console/threat-intelligence")({
  head: () => ({ meta: [{ title: "Threat Intelligence — TRIDENT" }] }),
  component: ThreatIntel,
});

const CONTACTS: Array<{
  id: string; category: string; level: ThreatLevel; bearing: string; range: string; action: string; note: string;
}> = [
  { id: "TGT-014", category: "SUBMARINE", level: "HIGH", bearing: "247°", range: "12.4 NM", action: "Alert tactical command and shadow contact.", note: "Acoustic profile matches diesel-electric class. Maintain track." },
  { id: "TGT-013", category: "TORPEDO", level: "CRITICAL", bearing: "081°", range: "3.1 NM", action: "Initiate evasive maneuver. Deploy countermeasures.", note: "Closing on bearing — confidence 96.4%." },
  { id: "TGT-012", category: "FRIGATE", level: "MEDIUM", bearing: "192°", range: "28.0 NM", action: "Continue passive observation, log signature.", note: "Friendly IFF unconfirmed, watch for posture change." },
  { id: "TGT-011", category: "CARGO_VESSEL", level: "LOW", bearing: "318°", range: "44.2 NM", action: "No action required.", note: "Civilian traffic crossing surveillance corridor." },
  { id: "TGT-010", category: "WHALE_POD", level: "LOW", bearing: "022°", range: "6.7 NM", action: "Log for biological intel feed.", note: "Cetacean vocalizations — not a threat." },
];

function ThreatIntel() {
  const counts = CONTACTS.reduce((a, c) => ({ ...a, [c.level]: (a[c.level] ?? 0) + 1 }), {} as Record<ThreatLevel, number>);
  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Doctrine" title="Threat Intelligence" description="Live target catalog with operational risk scoring and recommended actions." />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {(["LOW", "MEDIUM", "HIGH", "CRITICAL"] as ThreatLevel[]).map((lvl) => {
          const s = threatStyles(lvl);
          const Icon = lvl === "CRITICAL" ? AlertTriangle : lvl === "HIGH" ? ShieldAlert : lvl === "MEDIUM" ? Crosshair : ShieldCheck;
          return (
            <div key={lvl} className={cn("glass-panel rounded-lg p-4 ring-1", s.ring)}>
              <div className="flex items-center justify-between">
                <Icon className={cn("h-4 w-4", s.color)} strokeWidth={1.6} />
                <ThreatBadge level={lvl} size="sm" />
              </div>
              <div className="mt-3 mono text-3xl font-semibold tabular-nums text-foreground">{counts[lvl] ?? 0}</div>
              <div className="mono mt-1 text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{s.label}</div>
            </div>
          );
        })}
      </div>

      <div className="space-y-3">
        {CONTACTS.map((c) => {
          const s = threatStyles(c.level);
          return (
            <div key={c.id} className={cn("glass-panel rounded-md p-5 ring-1 transition hover:-translate-y-px", s.ring)}>
              <div className="flex flex-wrap items-center gap-4">
                <div className="mono text-[10px] uppercase tracking-[0.25em] text-cyan">{c.id}</div>
                <div className="text-base font-semibold tracking-wide">{c.category}</div>
                <ThreatBadge level={c.level} />
                <div className="ml-auto flex gap-5 mono text-[11px] text-muted-foreground">
                  <span>Bearing <span className="text-foreground">{c.bearing}</span></span>
                  <span>Range <span className="text-foreground">{c.range}</span></span>
                </div>
              </div>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div className={cn("rounded border border-transparent p-3 ring-1", s.bg, s.ring)}>
                  <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Recommended Action</div>
                  <div className="mt-1 text-sm">{c.action}</div>
                </div>
                <div className="rounded border border-border/60 bg-background/40 p-3">
                  <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Operational Notes</div>
                  <div className="mt-1 text-sm text-foreground/85">{c.note}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}