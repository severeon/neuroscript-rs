import React, { useState, useEffect } from 'react';
import init, { compile, analyze } from '@site/static/wasm/neuroscript.js';
import { STDLIB_BUNDLE } from './StdlibBundle';

/**
 * InteractiveExample - A component for tutorial pages that shows
 * NeuroScript code with live compilation and shape analysis.
 *
 * Props:
 *   - initialCode: The starting code to display
 *   - title: Optional title for the example
 *   - description: Optional description text
 *   - showAnalysis: Whether to show the analysis panel (default: true)
 *   - height: Height of the code editors (default: '300px')
 */
export default function InteractiveExample({
  initialCode,
  title,
  description,
  showAnalysis = true,
  height = '300px'
}) {
  const [input, setInput] = useState(initialCode || '');
  const [output, setOutput] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false);

  useEffect(() => {
    init().then(() => {
      setReady(true);
      handleCompile(initialCode || '');
    }).catch(e => {
      console.error("WASM init failed:", e);
      setError("Failed to initialize compiler: " + e);
    });
  }, []);

  const handleCompile = (source) => {
    if (!ready) return;
    try {
      // Prepend stdlib bundle and comment out 'use' statements
      const cleanSource = source.replace(/^use .*$/gm, '# $&');
      const fullSource = STDLIB_BUNDLE + '\n' + cleanSource;

      const result = compile(fullSource);
      setOutput(result);
      setError('');

      // Also run analysis if enabled
      if (showAnalysis) {
        try {
          const analysisJson = analyze(fullSource);
          setAnalysisData(JSON.parse(analysisJson));
        } catch (analysisError) {
          // Analysis errors don't block compilation
          setAnalysisData(null);
        }
      }
    } catch (e) {
      setError(e.toString());
      setOutput('');
      setAnalysisData(null);
    }
  };

  const handleChange = (e) => {
    const newVal = e.target.value;
    setInput(newVal);
    handleCompile(newVal);
  };

  return (
    <div className="interactive-example" style={{ marginBottom: '2rem' }}>
      {title && <h3 style={{ marginBottom: '0.5rem' }}>{title}</h3>}
      {description && <p style={{ marginBottom: '1rem', color: '#666' }}>{description}</p>}

      <div style={{ display: 'flex', gap: '1rem', minHeight: height }}>
        {/* Source Editor */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <strong style={{ fontSize: '0.9rem', color: '#666' }}>NeuroScript</strong>
          </div>
          <textarea
            value={input}
            onChange={handleChange}
            style={{
              flex: 1,
              fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
              padding: '1rem',
              resize: 'none',
              borderRadius: '8px',
              border: '1px solid #ccc',
              fontSize: '13px',
              lineHeight: '1.5',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4'
            }}
            spellCheck="false"
          />
        </div>

        {/* Output Panel */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <strong style={{ fontSize: '0.9rem', color: '#666' }}>PyTorch Output</strong>
            {showAnalysis && analysisData && (
              <button
                onClick={() => setShowAnalysisPanel(!showAnalysisPanel)}
                style={{
                  padding: '4px 8px',
                  fontSize: '0.8rem',
                  borderRadius: '4px',
                  border: '1px solid #ccc',
                  backgroundColor: showAnalysisPanel ? '#e0f0e0' : '#fff',
                  cursor: 'pointer'
                }}
              >
                {showAnalysisPanel ? 'Hide Analysis' : 'Show Analysis'}
              </button>
            )}
          </div>
          {error ? (
            <div style={{
              flex: 1,
              backgroundColor: '#fff0f0',
              color: '#d00',
              padding: '1rem',
              fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
              fontSize: '13px',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              borderRadius: '8px',
              border: '1px solid #ffcccc'
            }}>
              {error}
            </div>
          ) : (
            <textarea
              readOnly
              value={output}
              style={{
                flex: 1,
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
                padding: '1rem',
                backgroundColor: '#1e1e1e',
                color: '#d4d4d4',
                resize: 'none',
                borderRadius: '8px',
                border: '1px solid #333',
                fontSize: '13px',
                lineHeight: '1.5'
              }}
            />
          )}
        </div>
      </div>

      {/* Analysis Panel */}
      {showAnalysis && showAnalysisPanel && analysisData && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: '#f8f9fa',
          borderRadius: '8px',
          border: '1px solid #e0e0e0'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem' }}>Shape Analysis</h4>

          {/* Neurons */}
          {analysisData.neurons && analysisData.neurons.filter(n => !n.is_primitive).map((neuron, idx) => (
            <div key={idx} style={{ marginBottom: '1rem' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {neuron.name}
                {neuron.params.length > 0 && (
                  <span style={{ fontWeight: 'normal', color: '#666' }}>
                    ({neuron.params.map(p => p.name).join(', ')})
                  </span>
                )}
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'auto 1fr',
                gap: '0.25rem 1rem',
                fontSize: '0.9rem',
                paddingLeft: '1rem'
              }}>
                {neuron.inputs.map((port, i) => (
                  <React.Fragment key={`in-${i}`}>
                    <span style={{ color: '#666' }}>in{port.name !== 'default' ? ` ${port.name}` : ''}:</span>
                    <code style={{
                      backgroundColor: '#e8f5e9',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '0.85rem'
                    }}>{port.shape}</code>
                  </React.Fragment>
                ))}
                {neuron.outputs.map((port, i) => (
                  <React.Fragment key={`out-${i}`}>
                    <span style={{ color: '#666' }}>out{port.name !== 'default' ? ` ${port.name}` : ''}:</span>
                    <code style={{
                      backgroundColor: '#e3f2fd',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '0.85rem'
                    }}>{port.shape}</code>
                  </React.Fragment>
                ))}
              </div>

              {/* Connections */}
              {neuron.connections.length > 0 && (
                <div style={{ marginTop: '0.5rem', paddingLeft: '1rem' }}>
                  <span style={{ color: '#666', fontSize: '0.85rem' }}>Connections:</span>
                  <div style={{
                    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                    fontSize: '0.8rem',
                    color: '#444',
                    marginTop: '0.25rem'
                  }}>
                    {neuron.connections.map((conn, i) => (
                      <div key={i}>{conn.source} → {conn.destination}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Match Expressions */}
          {analysisData.match_expressions && analysisData.match_expressions.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h5 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>Match Expressions</h5>
              {analysisData.match_expressions.map((matchExpr, idx) => (
                <div key={idx} style={{ marginBottom: '0.5rem', paddingLeft: '1rem' }}>
                  <div style={{ fontSize: '0.85rem', color: '#666' }}>In {matchExpr.neuron}:</div>
                  {matchExpr.arms.map((arm, armIdx) => (
                    <div key={armIdx} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      fontSize: '0.85rem',
                      marginLeft: '1rem',
                      opacity: arm.is_reachable ? 1 : 0.5
                    }}>
                      <code style={{
                        backgroundColor: arm.is_reachable ? '#fff3e0' : '#f5f5f5',
                        padding: '2px 6px',
                        borderRadius: '3px'
                      }}>
                        {arm.pattern}
                        {arm.guard && <span style={{ color: '#666' }}> where {arm.guard}</span>}
                      </code>
                      {!arm.is_reachable && (
                        <span style={{ color: '#999', fontSize: '0.8rem' }}>(unreachable)</span>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
