import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/trident/AppShell";

export const Route = createFileRoute("/_console")({
  component: AppShell,
});