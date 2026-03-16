import type { ReactNode } from "react";

import type { EntityHit, SourceText } from "../types";

interface EntityHighlightProps {
  text: string;
  hits: EntityHit[];
  source: SourceText;
}

export function EntityHighlight({ text, hits, source }: EntityHighlightProps) {
  const sourceHits = hits
    .filter((hit) => hit.source_text === source)
    .sort((left, right) => left.start - right.start || left.end - right.end);

  const chunks: ReactNode[] = [];
  let cursor = 0;
  sourceHits.forEach((hit, index) => {
    const safeStart = Math.max(cursor, Math.max(0, hit.start));
    const safeEnd = Math.min(text.length, hit.end);
    if (safeEnd <= safeStart) {
      return;
    }
    if (safeStart > cursor) {
      chunks.push(<span key={`plain-${index}-${cursor}`}>{text.slice(cursor, safeStart)}</span>);
    }
    chunks.push(
      <mark key={`hit-${index}-${safeStart}`} className="highlight-hit">
        {text.slice(safeStart, safeEnd)}
      </mark>
    );
    cursor = safeEnd;
  });

  if (cursor < text.length) {
    chunks.push(<span key={`plain-tail-${cursor}`}>{text.slice(cursor)}</span>);
  }

  if (!chunks.length) {
    return <>{text}</>;
  }
  return <>{chunks}</>;
}
