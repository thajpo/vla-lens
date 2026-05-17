import type { ReactNode } from "react";

type PanelCardProps = {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
};

export function PanelCard({ title, actions, children }: PanelCardProps) {
  return (
    <section className="panel-card">
      <header>
        <h2>{title}</h2>
        {actions}
      </header>
      {children}
    </section>
  );
}
