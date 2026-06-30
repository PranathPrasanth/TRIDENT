import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { useState, type ReactNode } from "react";
import {
  LayoutDashboard,
  Activity,
  ShieldAlert,
  Eye,
  BarChart3,
  Info,
  Settings as SettingsIcon,
  Menu,
  ChevronLeft,
  Radar,
  Radio,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { UnderwaterBackground } from "./UnderwaterBackground";

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/acoustic-analysis", label: "Acoustic Analysis", icon: Activity },
  { to: "/threat-intelligence", label: "Threat Intelligence", icon: ShieldAlert },
  { to: "/explainability", label: "Explainability", icon: Eye },
  { to: "/model-statistics", label: "Model Statistics", icon: BarChart3 },
  { to: "/about", label: "About", icon: Info },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
] as const;

export function AppShell({ children }: { children?: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="relative flex min-h-screen text-foreground">
      <UnderwaterBackground intensity="subtle" />

      <aside
        className={cn(
          "sticky top-0 z-30 h-screen shrink-0 border-r border-sidebar-border bg-sidebar/70 backdrop-blur-xl transition-[width] duration-300",
          collapsed ? "w-[72px]" : "w-[252px]",
        )}
      >
        <div className="flex h-16 items-center justify-between px-4 border-b border-sidebar-border">
          <Link to="/" className="flex items-center gap-2.5 min-w-0">
            <div className="relative grid h-9 w-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-cyan/30 to-teal/10 glow-border">
              <Radar className="h-5 w-5 text-cyan" />
              <span className="absolute inset-0 rounded-md border border-cyan/40 animate-sonar" />
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <div className="text-sm font-semibold tracking-[0.22em] text-foreground">TRIDENT</div>
                <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Naval Intel</div>
              </div>
            )}
          </Link>
          <button
            onClick={() => setCollapsed((c) => !c)}
            aria-label="Toggle sidebar"
            className="grid h-7 w-7 place-items-center rounded text-muted-foreground hover:text-cyan hover:bg-secondary/60 transition"
          >
            {collapsed ? <Menu className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        <nav className="px-2 py-4 space-y-0.5">
          {NAV.map((item) => {
            const active = pathname === item.to || (item.to !== "/dashboard" && pathname.startsWith(item.to));
            const Icon = item.icon;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "group relative flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-all",
                  active
                    ? "bg-gradient-to-r from-cyan/15 to-transparent text-cyan"
                    : "text-sidebar-foreground/80 hover:text-foreground hover:bg-sidebar-accent/60",
                )}
              >
                {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r bg-cyan shadow-[0_0_12px_var(--cyan)]" />}
                <Icon className="h-[18px] w-[18px] shrink-0" strokeWidth={1.6} />
                {!collapsed && <span className="truncate tracking-wide">{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {!collapsed && (
          <div className="absolute bottom-4 left-3 right-3 rounded-md border border-sidebar-border bg-secondary/40 p-3">
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              <span className="size-1.5 rounded-full bg-threat-low animate-pulse" />
              System Online
            </div>
            <div className="mt-1 mono text-[11px] text-foreground/80">TRIDENT-v1.0.0</div>
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="flex-1 p-6 lg:p-8">
          <div className="mx-auto w-full max-w-[1600px] animate-fade-in">{children ?? <Outlet />}</div>
        </main>
        <TridentFooter />
      </div>
    </div>
  );
}

function TopBar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const current = NAV.find((n) => pathname.startsWith(n.to))?.label ?? "Overview";
  const time = new Date().toUTCString().split(" ").slice(0, 5).join(" ");
  return (
    <header className="sticky top-0 z-20 flex h-16 items-center gap-4 border-b border-border/60 bg-background/60 px-6 backdrop-blur-xl">
      <div className="flex items-center gap-3 min-w-0">
        <span className="mono text-[10px] uppercase tracking-[0.3em] text-muted-foreground">Sector / </span>
        <h1 className="truncate text-sm font-medium tracking-wide text-foreground">{current}</h1>
      </div>
      <div className="ml-auto flex items-center gap-5 text-[11px]">
        <div className="hidden md:flex items-center gap-2 text-muted-foreground">
          <Radio className="h-3.5 w-3.5 text-cyan" />
          <span className="mono">CHANNEL · SONAR-07</span>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-muted-foreground">
          <span className="size-1.5 rounded-full bg-threat-low animate-pulse" />
          <span className="mono uppercase tracking-widest">Operational</span>
        </div>
        <div className="mono text-muted-foreground">{time} UTC</div>
      </div>
    </header>
  );
}

export function TridentFooter() {
  return (
    <footer className="border-t border-border/60 bg-background/40 px-6 py-4 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] flex-col items-center justify-between gap-2 text-[11px] text-muted-foreground sm:flex-row">
        <div className="flex items-center gap-3">
          <span className="font-semibold tracking-[0.3em] text-foreground/80">TRIDENT</span>
          <span className="mono">v1.0.0</span>
        </div>
        <div className="flex items-center gap-5">
          <a href="https://github.com" target="_blank" rel="noreferrer" className="hover:text-cyan transition">GitHub</a>
          <span>MIT License</span>
          <span>Built with TensorFlow</span>
        </div>
      </div>
    </footer>
  );
}