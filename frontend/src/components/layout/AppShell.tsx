import type { ReactNode } from "react";
import { Database, FileText, LayoutDashboard, Microscope } from "lucide-react";

export type AppPage = "dataset" | "episode" | "interventions" | "probes";

type AppShellProps = {
  activePage: AppPage;
  onPageChange: (page: AppPage) => void;
  children: ReactNode;
};

export function AppShell({ activePage, onPageChange, children }: AppShellProps) {
  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand-block">
          <LayoutDashboard size={20} />
          <div>
            <strong>VLA-lens</strong>
          </div>
        </div>
        <nav className="top-nav" aria-label="Primary">
          <button
            className={activePage === "dataset" || activePage === "episode" ? "active" : ""}
            type="button"
            onClick={() => onPageChange("dataset")}
          >
            <Database size={16} />
            Dataset
          </button>
          <button
            className={activePage === "probes" ? "active" : ""}
            type="button"
            onClick={() => onPageChange("probes")}
          >
            <Microscope size={16} />
            Probes
          </button>
          <button
            className={activePage === "interventions" ? "active" : ""}
            type="button"
            onClick={() => onPageChange("interventions")}
          >
            <FileText size={16} />
            Interventions
          </button>
        </nav>
      </header>
      {children}
    </div>
  );
}
