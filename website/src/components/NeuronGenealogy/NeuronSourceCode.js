import React from 'react';
import CodeBlock from '@theme/CodeBlock';

export default function NeuronSourceCode({ source, sourceFile }) {
  if (!source) return null;

  return (
    <details className="ng-source-section">
      <summary className="ng-source-toggle">Source Code</summary>
      {sourceFile && <div className="ng-source-path">{sourceFile}</div>}
      <CodeBlock language="python">{source}</CodeBlock>
    </details>
  );
}
