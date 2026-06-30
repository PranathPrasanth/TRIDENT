import { createFileRoute } from "@tanstack/react-router";
import { Panel, SectionTitle } from "@/components/trident/primitives";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  BarChart, Bar, AreaChart, Area,
} from "recharts";

export const Route = createFileRoute("/_console/model-statistics")({
  head: () => ({ meta: [{ title: "Model Statistics — TRIDENT" }] }),
  component: ModelStats,
});

const loss = Array.from({ length: 40 }).map((_, i) => ({
  epoch: i + 1,
  train: +(1.6 * Math.exp(-i / 10) + 0.08 + Math.random() * 0.03).toFixed(3),
  val: +(1.6 * Math.exp(-i / 9) + 0.14 + Math.random() * 0.04).toFixed(3),
}));

const acc = Array.from({ length: 40 }).map((_, i) => ({
  epoch: i + 1,
  train: +(0.4 + 0.58 * (1 - Math.exp(-i / 8)) + Math.random() * 0.01).toFixed(3),
  val: +(0.4 + 0.55 * (1 - Math.exp(-i / 8.5)) + Math.random() * 0.015).toFixed(3),
}));

const CLASSES = ["Submarine", "Torpedo", "Frigate", "Cargo", "Whale", "Ambient"];
const matrix = [
  [182, 1, 2, 0, 0, 1],
  [3, 174, 1, 0, 0, 0],
  [2, 1, 168, 4, 0, 3],
  [0, 0, 5, 191, 0, 2],
  [0, 0, 0, 0, 156, 4],
  [1, 0, 2, 3, 5, 178],
];

const classDist = CLASSES.map((c, i) => ({ name: c, samples: [2470, 2310, 2580, 2740, 1980, 2740][i] }));

const axisStyle = { stroke: "oklch(0.55 0.03 230)", fontSize: 11 };
const gridStyle = { stroke: "oklch(0.35 0.04 230 / 0.3)" };
const tooltipStyle = {
  background: "oklch(0.16 0.03 245)",
  border: "1px solid oklch(0.4 0.06 220 / 0.4)",
  borderRadius: 6,
  fontFamily: "JetBrains Mono",
  fontSize: 12,
};

function ModelStats() {
  return (
    <div className="space-y-6">
      <SectionTitle eyebrow="Model Telemetry" title="Model Statistics" description="Architecture, training dynamics, and class-level performance for the TRIDENT-CNN classifier." />

      <div className="grid gap-4 md:grid-cols-4">
        {[
          ["Architecture", "TRIDENT-CNN v3"],
          ["CNN Layers", "12"],
          ["Parameters", "2.74 M"],
          ["Batch Size", "64"],
          ["Epochs", "40"],
          ["Learning Rate", "3e-4"],
          ["Train Accuracy", "98.94%"],
          ["Val Accuracy", "97.82%"],
        ].map(([k, v]) => (
          <div key={k} className="glass-panel rounded-lg p-4">
            <div className="mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{k}</div>
            <div className="mono mt-2 text-xl font-semibold text-foreground">{v}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Loss Curves">
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={loss}>
                <CartesianGrid {...gridStyle} strokeDasharray="3 3" />
                <XAxis dataKey="epoch" {...axisStyle} />
                <YAxis {...axisStyle} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
                <Line type="monotone" dataKey="train" stroke="oklch(0.82 0.15 200)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val" stroke="oklch(0.78 0.16 85)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title="Accuracy Curves">
          <div className="h-72">
            <ResponsiveContainer>
              <AreaChart data={acc}>
                <defs>
                  <linearGradient id="a1" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.82 0.15 200)" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="oklch(0.82 0.15 200)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...gridStyle} strokeDasharray="3 3" />
                <XAxis dataKey="epoch" {...axisStyle} />
                <YAxis domain={[0.3, 1]} {...axisStyle} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
                <Area type="monotone" dataKey="train" stroke="oklch(0.82 0.15 200)" fill="url(#a1)" strokeWidth={2} />
                <Area type="monotone" dataKey="val" stroke="oklch(0.7 0.12 185)" fill="transparent" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Confusion Matrix" className="lg:col-span-2">
          <ConfusionMatrix />
        </Panel>

        <Panel title="Dataset Class Distribution" className="lg:col-span-2">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={classDist}>
                <CartesianGrid {...gridStyle} strokeDasharray="3 3" />
                <XAxis dataKey="name" {...axisStyle} />
                <YAxis {...axisStyle} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "oklch(0.3 0.05 220 / 0.2)" }} />
                <Bar dataKey="samples" fill="oklch(0.78 0.14 200)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function ConfusionMatrix() {
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 mono text-xs">
        <thead>
          <tr>
            <th className="p-2" />
            {CLASSES.map((c) => (
              <th key={c} className="p-2 text-[10px] uppercase tracking-wider text-muted-foreground">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-right text-[10px] uppercase tracking-wider text-muted-foreground">{CLASSES[i]}</td>
              {row.map((v, j) => {
                const t = v / max;
                return (
                  <td
                    key={j}
                    className="h-10 w-14 rounded text-center align-middle text-[11px]"
                    style={{
                      background: `oklch(0.78 0.14 200 / ${0.1 + t * 0.7})`,
                      color: t > 0.4 ? "oklch(0.13 0.03 245)" : "oklch(0.9 0.02 220)",
                      fontWeight: i === j ? 600 : 400,
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}