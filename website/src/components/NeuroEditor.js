import React, { useState, useEffect, useCallback, useRef } from 'react';
import init, { compile, analyze, list_neurons } from '@site/static/wasm/neuroscript.js';
import { STDLIB_BUNDLE } from './StdlibBundle';

const MONO_FONT = '"Fira Code", "Cascadia Code", "SF Mono", Monaco, "Inconsolata", "Roboto Mono", "Source Code Pro", Menlo, Consolas, "DejaVu Sans Mono", monospace';

/**
 * NeuroEditor - Unified component for tutorial examples and playground.
 *
 * mode='tutorial' (default): compact editor with optional analysis panel.
 * mode='playground': full-featured editor with example selector, neuron picker, etc.
 *
 * Individual boolean props override mode defaults.
 */
export default function NeuroEditor({
  mode = 'tutorial',
  initialCode = '',
  title,
  description,
  showAnalysis,
  showExampleSelector,
  showNeuronSelector,
  showCompileButton,
  showCopyButton,
  showStats,
  height,
  responsive,
  examples: examplesProp,
  defaultExampleId = 'mlp',
}) {
  const isPlayground = mode === 'playground';

  // Resolve feature flags: explicit prop > mode default
  const feat = {
    analysis:        showAnalysis        ?? !isPlayground,
    exampleSelector: showExampleSelector ?? isPlayground,
    neuronSelector:  showNeuronSelector  ?? isPlayground,
    compileButton:   showCompileButton   ?? true,
    copyButton:      showCopyButton      ?? isPlayground,
    stats:           showStats           ?? isPlayground,
    responsive:      responsive          ?? isPlayground,
  };
  const editorHeight = height || (isPlayground ? '500px' : '300px');

  // -- Lazy-load playground examples only when needed --
  const examplesRef = useRef(null);
  const categoriesRef = useRef(null);
  if (feat.exampleSelector && !examplesRef.current) {
    // Dynamic require so tutorial pages never load PlaygroundExamples
    const mod = require('./PlaygroundExamples');
    examplesRef.current = examplesProp || mod.EXAMPLES;
    categoriesRef.current = mod.getExamplesByCategory();
  }
  const EXAMPLES = examplesRef.current || [];
  const examplesByCategory = categoriesRef.current || {};

  // -- State --
  const [input, setInput]                     = useState(initialCode);
  const [output, setOutput]                   = useState('');
  const [error, setError]                     = useState('');
  const [ready, setReady]                     = useState(false);
  const [selectedExample, setSelectedExample] = useState(defaultExampleId);
  const [availableNeurons, setAvailableNeurons] = useState([]);
  const [selectedNeuron, setSelectedNeuron]   = useState(null);
  const [analysisData, setAnalysisData]       = useState(null);
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false);
  const [copied, setCopied]                   = useState(false);
  const [compiling, setCompiling]             = useState(false);
  const [isMobile, setIsMobile]               = useState(false);

  const compileTimerRef = useRef(null);

  // -- Compilation --
  const handleCompile = useCallback((source, neuronName = null) => {
    if (!ready) return;
    setCompiling(true);
    try {
      const cleanSource = source.replace(/^use .*$/gm, '# $&');
      const fullSource = STDLIB_BUNDLE + '\n' + cleanSource;

      // Neuron list (playground)
      if (feat.neuronSelector) {
        const neuronsJson = list_neurons(fullSource);
        const neurons = JSON.parse(neuronsJson);
        setAvailableNeurons(neurons);
        if (!neuronName && neurons.length === 1) {
          neuronName = neurons[0];
          setSelectedNeuron(neurons[0]);
        }
      }

      const result = compile(fullSource, neuronName || undefined);
      setOutput(result);
      setError('');

      // Analysis (tutorial)
      if (feat.analysis) {
        try {
          const analysisJson = analyze(fullSource);
          setAnalysisData(JSON.parse(analysisJson));
        } catch (_) {
          setAnalysisData(null);
        }
      }
    } catch (e) {
      setError(e.toString());
      setOutput('');
      setAvailableNeurons([]);
      setAnalysisData(null);
    } finally {
      setCompiling(false);
    }
  }, [ready, feat.neuronSelector, feat.analysis]);

  // -- Debounce --
  const debouncedCompile = useCallback((source, neuronName) => {
    if (compileTimerRef.current) clearTimeout(compileTimerRef.current);
    compileTimerRef.current = setTimeout(() => handleCompile(source, neuronName), 500);
  }, [handleCompile]);

  // -- Load example (playground) --
  const loadExample = useCallback((example) => {
    setInput(example.code);
    setSelectedExample(example.id);
    setSelectedNeuron(example.targetNeuron);
    handleCompile(example.code, example.targetNeuron);
  }, [handleCompile]);

  // -- WASM init --
  useEffect(() => {
    init().then(() => setReady(true)).catch(e => {
      console.error('WASM init failed:', e);
      setError('Failed to initialize compiler: ' + e);
    });
  }, []);

  // -- Initial compile --
  useEffect(() => {
    if (!ready) return;
    if (feat.exampleSelector) {
      const defaultEx = EXAMPLES.find(ex => ex.id === defaultExampleId);
      if (defaultEx) loadExample(defaultEx);
    } else {
      handleCompile(initialCode);
    }
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Mobile detect --
  useEffect(() => {
    if (!feat.responsive) return;
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, [feat.responsive]);

  // -- Event handlers --
  const handleChange = (e) => {
    const val = e.target.value;
    setInput(val);
    debouncedCompile(val, selectedNeuron);
  };

  const handleExampleChange = (e) => {
    const ex = EXAMPLES.find(x => x.id === e.target.value);
    if (ex) loadExample(ex);
  };

  const handleNeuronChange = (e) => setSelectedNeuron(e.target.value);

  const handleManualCompile = () => handleCompile(input, selectedNeuron);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  };

  const currentExample = feat.exampleSelector
    ? EXAMPLES.find(ex => ex.id === selectedExample)
    : null;

  // ==========================================================================
  // Render
  // ==========================================================================
  return (
    <div style={isPlayground
      ? { display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '100%' }
      : { marginBottom: '2rem' }
    }>
      {/* Tutorial header */}
      {title && <h3 style={{ marginBottom: '0.5rem' }}>{title}</h3>}
      {description && <p style={{ marginBottom: '1rem', color: '#666' }}>{description}</p>}

      {/* Playground toolbar */}
      {(feat.exampleSelector || feat.compileButton) && (
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {feat.exampleSelector && (
            <div style={{ flex: '1', minWidth: '250px' }}>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                Example:
              </label>
              <select
                value={selectedExample}
                onChange={handleExampleChange}
                disabled={!ready}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '4px',
                  border: '1px solid var(--ifm-color-emphasis-300)',
                  backgroundColor: 'var(--ifm-background-color)',
                  color: 'var(--ifm-font-color-base)',
                  fontSize: '14px',
                }}
              >
                {Object.entries(examplesByCategory).map(([category, examples]) => (
                  <optgroup key={category} label={category}>
                    {examples.map(ex => (
                      <option key={ex.id} value={ex.id}>{ex.title}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          )}

          {feat.neuronSelector && availableNeurons.length > 1 && (
            <div style={{ minWidth: '200px' }}>
              <label style={{ display: 'block', marginBottom: '0.25rem', fontWeight: 'bold' }}>
                Compile:
              </label>
              <select
                value={selectedNeuron || ''}
                onChange={handleNeuronChange}
                disabled={!ready}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '4px',
                  border: '1px solid var(--ifm-color-emphasis-300)',
                  backgroundColor: 'var(--ifm-background-color)',
                  color: 'var(--ifm-font-color-base)',
                  fontSize: '14px',
                }}
              >
                <option value="">Auto-detect</option>
                {availableNeurons.map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          )}

          {feat.compileButton && (
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button
                onClick={handleManualCompile}
                disabled={!ready || compiling}
                className="button button--primary"
                style={{ fontSize: '14px' }}
              >
                {compiling ? '\u23F3 Compiling...' : '\u25B6\uFE0F Compile'}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Playground example description + features */}
      {currentExample && (
        <div style={{
          padding: '0.75rem',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          borderRadius: '8px',
          borderLeft: '4px solid var(--ifm-color-primary)',
        }}>
          <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold' }}>
            {currentExample.description}
          </p>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {currentExample.features.map(f => (
              <span key={f} style={{
                padding: '0.25rem 0.5rem',
                backgroundColor: 'var(--ifm-color-primary)',
                color: 'white',
                borderRadius: '12px',
                fontSize: '12px',
                fontWeight: '500',
              }}>{f}</span>
            ))}
          </div>
        </div>
      )}

      {/* Side-by-side editors */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        minHeight: editorHeight,
        flexDirection: feat.responsive && isMobile ? 'column' : 'row',
      }}>
        {/* Source */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem',
          }}>
            {isPlayground
              ? <h3 style={{ margin: 0 }}>Source (NeuroScript)</h3>
              : <strong style={{ fontSize: '0.9rem', color: '#666' }}>NeuroScript</strong>}
            {compiling && isPlayground && (
              <span style={{ fontSize: '12px', color: 'var(--ifm-color-primary)' }}>
                Compiling...
              </span>
            )}
          </div>
          <textarea
            value={input}
            onChange={handleChange}
            disabled={isPlayground && !ready}
            spellCheck="false"
            style={{
              flex: 1,
              fontFamily: MONO_FONT,
              padding: '1rem',
              resize: 'none',
              borderRadius: '8px',
              fontSize: '13px',
              lineHeight: isPlayground ? '1.6' : '1.5',
              ...(isPlayground ? {
                border: '1px solid var(--ifm-color-emphasis-300)',
                backgroundColor: 'var(--ifm-background-color)',
                color: 'var(--ifm-font-color-base)',
                minHeight: '400px',
              } : {
                border: '1px solid #ccc',
                backgroundColor: '#1e1e1e',
                color: '#d4d4d4',
              }),
            }}
          />
        </div>

        {/* Output */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem',
          }}>
            {isPlayground
              ? <h3 style={{ margin: 0 }}>Output (PyTorch)</h3>
              : <strong style={{ fontSize: '0.9rem', color: '#666' }}>PyTorch Output</strong>}
            {feat.copyButton && output && (
              <button
                onClick={handleCopy}
                className="button button--sm button--secondary"
                style={{ fontSize: '12px' }}
              >
                {copied ? '\u2713 Copied!' : '\uD83D\uDCCB Copy'}
              </button>
            )}
            {feat.analysis && analysisData && !isPlayground && (
              <button
                onClick={() => setShowAnalysisPanel(!showAnalysisPanel)}
                style={{
                  padding: '4px 8px',
                  fontSize: '0.8rem',
                  borderRadius: '4px',
                  border: '1px solid #ccc',
                  backgroundColor: showAnalysisPanel ? '#e0f0e0' : '#fff',
                  cursor: 'pointer',
                }}
              >
                {showAnalysisPanel ? 'Hide Analysis' : 'Show Analysis'}
              </button>
            )}
          </div>

          {error ? (
            <div style={{
              flex: 1,
              padding: '1rem',
              fontFamily: MONO_FONT,
              fontSize: '13px',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              borderRadius: '8px',
              lineHeight: isPlayground ? '1.6' : '1.5',
              ...(isPlayground ? {
                backgroundColor: 'var(--ifm-color-danger-contrast-background)',
                color: 'var(--ifm-color-danger-contrast-foreground)',
                border: '1px solid var(--ifm-color-danger)',
                minHeight: '400px',
              } : {
                backgroundColor: '#fff0f0',
                color: '#d00',
                border: '1px solid #ffcccc',
              }),
            }}>
              {error}
            </div>
          ) : (
            <div style={{
              flex: 1,
              fontFamily: MONO_FONT,
              padding: '1rem',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              overflow: 'auto',
              borderRadius: '8px',
              border: '1px solid #333',
              whiteSpace: 'pre',
              fontSize: '13px',
              lineHeight: isPlayground ? '1.6' : '1.5',
              ...(isPlayground ? { minHeight: '400px' } : {}),
            }}>
              {output || (isPlayground ? 'Compiled PyTorch code will appear here...' : '')}
            </div>
          )}
        </div>
      </div>

      {/* Analysis panel (tutorial) */}
      {feat.analysis && showAnalysisPanel && analysisData && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: '#f8f9fa',
          borderRadius: '8px',
          border: '1px solid #e0e0e0',
        }}>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem' }}>Shape Analysis</h4>

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
                paddingLeft: '1rem',
              }}>
                {neuron.inputs.map((port, i) => (
                  <React.Fragment key={`in-${i}`}>
                    <span style={{ color: '#666' }}>in{port.name !== 'default' ? ` ${port.name}` : ''}:</span>
                    <code style={{
                      backgroundColor: '#e8f5e9',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '0.85rem',
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
                      fontSize: '0.85rem',
                    }}>{port.shape}</code>
                  </React.Fragment>
                ))}
              </div>

              {neuron.connections.length > 0 && (
                <div style={{ marginTop: '0.5rem', paddingLeft: '1rem' }}>
                  <span style={{ color: '#666', fontSize: '0.85rem' }}>Connections:</span>
                  <div style={{
                    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                    fontSize: '0.8rem',
                    color: '#444',
                    marginTop: '0.25rem',
                  }}>
                    {neuron.connections.map((conn, i) => (
                      <div key={i}>{conn.source} → {conn.destination}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

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
                      opacity: arm.is_reachable ? 1 : 0.5,
                    }}>
                      <code style={{
                        backgroundColor: arm.is_reachable ? '#fff3e0' : '#f5f5f5',
                        padding: '2px 6px',
                        borderRadius: '3px',
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

      {/* Stats footer (playground) */}
      {feat.stats && output && (
        <div style={{
          padding: '0.5rem',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          borderRadius: '4px',
          fontSize: '12px',
          color: 'var(--ifm-color-emphasis-700)',
          textAlign: 'right',
        }}>
          {availableNeurons.length} neuron{availableNeurons.length !== 1 ? 's' : ''} • {output.split('\n').length} lines generated
        </div>
      )}
    </div>
  );
}
