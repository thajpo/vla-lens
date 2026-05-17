import type { ResolvedSelection } from "../../types/workbench";
import { PanelCard } from "../layout/PanelCard";

type ConfusionMatrixPanelProps = {
  resolution?: ResolvedSelection | null;
};

export function ConfusionMatrixPanel({ resolution }: ConfusionMatrixPanelProps) {
  const rows = ((resolution?.target_object_cell?.confusion_matrix as Record<string, unknown>[]) ?? [])
    .slice(0, 30);
  return (
    <PanelCard title="Confusion Matrix">
      {!rows.length ? <div className="empty-state">No confusion matrix for this cell.</div> : null}
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Actual</th>
              <th>Predicted</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${String(row.actual)}-${String(row.predicted)}-${index}`}>
                <td>{String(row.actual ?? "")}</td>
                <td>{String(row.predicted ?? "")}</td>
                <td>{String(row.count ?? "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelCard>
  );
}
