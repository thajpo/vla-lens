import { useId } from "react";
import type { InfoHoverCardData } from "./infoHoverModel";

type InfoHoverCardProps = {
  card: InfoHoverCardData;
  className?: string;
  id?: string;
};

type InfoIconTriggerProps = {
  card: InfoHoverCardData;
  className?: string;
  label?: string;
};

type InlineInfoTextProps = {
  card: InfoHoverCardData;
  className?: string;
  label: string;
};

export function InfoIconTrigger({
  card,
  className = "",
  label,
}: InfoIconTriggerProps) {
  const tooltipId = useId();
  return (
    <span className={["info-hover-trigger info-icon-trigger", className].filter(Boolean).join(" ")}>
      <button
        aria-describedby={tooltipId}
        aria-label={label ?? `About ${card.title}`}
        className="info-hover-button info-icon-button"
        type="button"
      >
        i
      </button>
      <InfoHoverCard card={card} id={tooltipId} />
    </span>
  );
}

export function InlineInfoText({
  card,
  className = "",
  label,
}: InlineInfoTextProps) {
  const tooltipId = useId();
  return (
    <span className={["info-hover-trigger info-inline-trigger", className].filter(Boolean).join(" ")}>
      <span
        aria-describedby={tooltipId}
        className="info-hover-button info-inline-button"
        tabIndex={0}
      >
        {label}
      </span>
      <InfoHoverCard card={card} id={tooltipId} />
    </span>
  );
}

export function InfoHoverCard({ card, className = "", id }: InfoHoverCardProps) {
  return (
    <span className={["info-hover-card", className].filter(Boolean).join(" ")} id={id} role="tooltip">
      <span className="info-hover-card-title">{card.title}</span>
      {card.subtitle ? <span className="info-hover-card-subtitle">{card.subtitle}</span> : null}
      {card.groups.map((group) => (
        <span className="info-hover-card-group" key={group.title}>
          <span className="info-hover-card-group-title">{group.title}</span>
          {group.lines.map((line) => (
            <span className="info-hover-card-line" key={`${group.title}-${line.label}`}>
              <span>{line.label}</span>
              <span>{line.value}</span>
            </span>
          ))}
        </span>
      ))}
    </span>
  );
}
