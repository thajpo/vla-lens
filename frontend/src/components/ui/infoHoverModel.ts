export type InfoHoverLine = {
  label: string;
  value: string;
};

export type InfoHoverGroup = {
  lines: InfoHoverLine[];
  title: string;
};

export type InfoHoverCardData = {
  groups: InfoHoverGroup[];
  subtitle?: string;
  title: string;
};

export function infoTextCard(title: string, value: string): InfoHoverCardData {
  return {
    groups: [
      {
        lines: [{ label: "Meaning", value }],
        title: "Info",
      },
    ],
    title,
  };
}
