import type { ResolvedSelection } from "../../types/workbench";
import { PanelCard } from "../layout/PanelCard";

type ExamplesPanelProps = {
  resolution?: ResolvedSelection | null;
};

export function ExamplesPanel({ resolution }: ExamplesPanelProps) {
  const rows = Object.values(resolution?.examples ?? {})
    .flat()
    .slice(0, 20);
  return (
    <PanelCard title="Examples">
      {!rows.length ? <div className="empty-state">No linked examples.</div> : null}
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Episode</th>
              <th>Timestep</th>
              <th>Actual</th>
              <th>Predicted</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${String(row.example_id ?? row.trace_id)}-${index}`}>
                <td>{String(row.trace_id ?? row.episode_id ?? "")}</td>
                <td>{String(row.timestep ?? "")}</td>
                <td>{String(row.actual ?? row.target_object ?? "")}</td>
                <td>{String(row.predicted ?? "")}</td>
                <td>{String(row.prediction_status ?? "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelCard>
  );
}
