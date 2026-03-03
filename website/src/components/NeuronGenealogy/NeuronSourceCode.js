import React, { useState } from 'react';
import CodeBlock from '@theme/CodeBlock';

export default function NeuronSourceCode({ source, sourceFile }) {
  const [expanded, setExpanded] = useState(false);

  if (!source) return null;

  return (
    <div className="ng-source-section">
      <h3
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer', userSelect: 'none' }}
      >
        {expanded ? '\u25BE' : '\u25B8'} Source Code
      </h3>
      {sourceFile && <div className="ng-source-path">{sourceFile}</div>}
      {expanded && (
        <CodeBlock language="neuroscript">{source}</CodeBlock>
      )}
    </div>
  );
}
