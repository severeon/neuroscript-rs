import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import init, { compile, analyze, list_neurons } from '@site/static/wasm/neuroscript.js';

// Lazy-load Monaco — if CDN fails, editors fall back to plain textareas
let Editor;
let registerNeuroScript;
try {
  Editor = require('@monaco-editor/react').default;
  registerNeuroScript = require('../neuroscript-monaco-setup').registerNeuroScript;
} catch (_) {
  // Will be caught at render time via editorFailed state
}

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
  layout,
  initialCode = '',
  title,
  description,
  showAnalysis,
  showExampleSelector,
  showCompileButton,
  showCopyButton,
  showStats,
  height,
  responsive,
  examples: examplesProp,
  defaultExampleId = 'mlp',
}) {
  const isPlayground = mode === 'playground';
  const { colorMode } = useColorMode();
  const isDark = colorMode === 'dark';

  // Layout: 'vertical' stacks panels top/bottom (full width each),
  //         'horizontal' places them side-by-side (current default for playground)
  const resolvedLayout = layout || (isPlayground ? 'horizontal' : 'vertical');
  const isVertical = resolvedLayout === 'vertical';

  // Resolve feature flags: explicit prop > mode default
  const feat = {
    analysis:        showAnalysis        ?? !isPlayground,
    exampleSelector: showExampleSelector ?? isPlayground,
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
  const [analysisData, setAnalysisData]       = useState(null);
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false);
  const [copied, setCopied]                   = useState(false);
  const [compiling, setCompiling]             = useState(false);
  const [isMobile, setIsMobile]               = useState(false);
  const [editorFailed, setEditorFailed]       = useState(!Editor);

  const compileTimerRef = useRef(null);

  // -- Compilation --
  const handleCompile = useCallback((source, neuronName = null) => {
    if (!ready) return;
    setCompiling(true);
    try {
      const cleanSource = source.replace(/^use .*$/gm, '# $&');

      // Auto-detect neuron: list all, pick single neuron automatically
      try {
        const neuronsJson = list_neurons(cleanSource);
        const neurons = JSON.parse(neuronsJson);
        setAvailableNeurons(neurons);
        if (!neuronName && neurons.length === 1) {
          neuronName = neurons[0];
        }
      } catch (_) {
        setAvailableNeurons([]);
      }

      const result = compile(cleanSource, neuronName || undefined);
      setOutput(result);
      setError('');

      // Analysis (tutorial)
      if (feat.analysis) {
        try {
          const analysisJson = analyze(cleanSource);
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
  }, [ready, feat.analysis]);

  // -- Debounce --
  const debouncedCompile = useCallback((source, neuronName) => {
    if (compileTimerRef.current) clearTimeout(compileTimerRef.current);
    compileTimerRef.current = setTimeout(() => handleCompile(source, neuronName), 500);
  }, [handleCompile]);

  // -- Load example (playground) --
  const loadExample = useCallback((example) => {
    setInput(example.code);
    setSelectedExample(example.id);
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
  const handleEditorChange = (value) => {
    const val = value || '';
    setInput(val);
    debouncedCompile(val);
  };

  const handleExampleChange = (e) => {
    const ex = EXAMPLES.find(x => x.id === e.target.value);
    if (ex) loadExample(ex);
  };

  const handleManualCompile = () => handleCompile(input);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  };

  const handleEditorBeforeMount = (monaco) => {
    if (registerNeuroScript) registerNeuroScript(monaco);
  };

  // If Monaco module failed to load, fall back immediately
  useEffect(() => {
    if (!Editor) setEditorFailed(true);
  }, []);

  const currentExample = feat.exampleSelector
    ? EXAMPLES.find(ex => ex.id === selectedExample)
    : null;

  // Shared Monaco options
  const sharedEditorOptions = {
    fontFamily: MONO_FONT,
    fontSize: isPlayground ? 13 : 12,
    lineHeight: isPlayground ? 1.6 : 1.5,
    minimap: { enabled: false },
    automaticLayout: true,
    scrollBeyondLastLine: false,
    tabSize: 4,
    renderLineHighlight: 'none',
    overviewRulerLanes: 0,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
    scrollbar: {
      verticalScrollbarSize: 8,
      horizontalScrollbarSize: 8,
    },
    padding: { top: 8, bottom: 8 },
  };

  const sourceEditorHeight = isVertical
    ? `${Math.max((input.split('\n').length + 1) * (isPlayground ? 21 : 18), isPlayground ? 400 : 72)}px`
    : editorHeight;

  const monacoTheme = isDark ? 'neuroscript-dark' : 'neuroscript-light';

  // ==========================================================================
  // Render
  // ==========================================================================
  return (
    <div style={isPlayground
      ? { display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '100%' }
      : { marginBottom: '2rem' }
    }>
      {/* Tutorial header */}
      {title && <h3 style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>{title}</h3>}
      {description && <p style={{ marginBottom: '1rem', color: 'var(--ifm-color-emphasis-600)' }}>{description}</p>}

      {/* Playground toolbar */}
      {(feat.exampleSelector || feat.compileButton) && (
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
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

          {feat.compileButton && (
            <div style={{ marginLeft: 'auto' }}>
              <button
                onClick={handleManualCompile}
                disabled={!ready || compiling}
                className="button button--outline button--primary"
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

      {/* Editor panels */}
      <div style={{
        display: 'flex',
        gap: isVertical ? '0.5rem' : '1rem',
        ...(isVertical
          ? { flexDirection: 'column' }
          : {
              minHeight: editorHeight,
              flexDirection: feat.responsive && isMobile ? 'column' : 'row',
            }),
      }}>
        {/* Source */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          ...(isVertical ? {} : { flex: 1 }),
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.25rem',
          }}>
            {isPlayground
              ? <h3 style={{ margin: 0, marginTop: '1rem' }}>Source (NeuroScript)</h3>
              : <strong style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-600)' }}>NeuroScript</strong>}
            {compiling && isPlayground && (
              <span style={{ fontSize: '12px', color: 'var(--ifm-color-primary)' }}>
                Compiling...
              </span>
            )}
          </div>
          <div style={{
            border: '1px solid var(--ifm-color-emphasis-300)',
            borderRadius: '8px',
            overflow: 'hidden',
            ...(isVertical ? {} : { flex: 1 }),
          }}>
            {editorFailed ? (
              <textarea
                value={input}
                onChange={(e) => { setInput(e.target.value); debouncedCompile(e.target.value); }}
                spellCheck={false}
                style={{
                  width: '100%',
                  height: sourceEditorHeight,
                  fontFamily: MONO_FONT,
                  fontSize: isPlayground ? '13px' : '12px',
                  lineHeight: isPlayground ? '1.6' : '1.5',
                  padding: '8px',
                  border: 'none',
                  resize: 'vertical',
                  backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
                  color: isDark ? '#d4d4d4' : '#24292e',
                  boxSizing: 'border-box',
                }}
              />
            ) : (
              <Editor
                height={sourceEditorHeight}
                language="neuroscript"
                theme={monacoTheme}
                value={input}
                onChange={handleEditorChange}
                beforeMount={handleEditorBeforeMount}
                onMount={() => {}}
                loading={<div style={{ padding: '1rem', fontFamily: MONO_FONT, fontSize: '12px', color: 'var(--ifm-color-emphasis-500)' }}>Loading editor...</div>}
                options={{
                  ...sharedEditorOptions,
                  readOnly: isPlayground && !ready,
                  lineNumbers: isPlayground ? 'on' : 'off',
                }}
              />
            )}
          </div>
        </div>

        {/* Output */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          ...(isVertical ? { marginTop: '0.5rem' } : { flex: 1 }),
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.25rem',
          }}>
            {isPlayground
              ? <h3 style={{ margin: 0, marginTop: '1rem' }}>Output (PyTorch)</h3>
              : <strong style={{ fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-600)' }}>PyTorch Output</strong>}
            {feat.copyButton && output && (
              <button
                onClick={handleCopy}
                className="button button--sm button--outline button--primary"
                style={{ fontSize: '12px' }}
              >
                {copied ? '\u2713 Copied!' : '\uD83D\uDCCB Copy'}
              </button>
            )}
          </div>

          {error ? (
            <div style={{
              padding: '0.75rem',
              fontFamily: MONO_FONT,
              fontSize: isPlayground ? '13px' : '12px',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              borderRadius: '8px',
              lineHeight: isPlayground ? '1.6' : '1.5',
              ...(isVertical
                ? { maxHeight: '400px' }
                : { flex: 1 }),
              ...(isPlayground ? {
                backgroundColor: 'var(--ifm-color-danger-contrast-background)',
                color: 'var(--ifm-color-danger-contrast-foreground)',
                border: '1px solid var(--ifm-color-danger)',
                minHeight: '400px',
              } : {
                backgroundColor: 'var(--ifm-color-danger-contrast-background)',
                color: 'var(--ifm-color-danger)',
                border: '1px solid var(--ifm-color-danger)',
              }),
            }}>
              {error}
            </div>
          ) : (
            <div style={{
              border: '1px solid var(--ifm-color-emphasis-300)',
              borderRadius: '8px',
              overflow: 'hidden',
              ...(isVertical
                ? { maxHeight: '400px' }
                : { flex: 1 }),
            }}>
              {editorFailed ? (
                <pre style={{
                  margin: 0,
                  padding: '8px',
                  fontFamily: MONO_FONT,
                  fontSize: isPlayground ? '13px' : '12px',
                  lineHeight: isPlayground ? '1.6' : '1.5',
                  height: isVertical ? 'auto' : editorHeight,
                  overflow: 'auto',
                  backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
                  color: isDark ? '#d4d4d4' : '#24292e',
                }}>{output || (isPlayground ? '# Compiled PyTorch code will appear here...' : '')}</pre>
              ) : (
                <Editor
                  height={isVertical
                    ? `${Math.max((output.split('\n').length + 1) * (isPlayground ? 21 : 18), isPlayground ? 400 : 72)}px`
                    : editorHeight}
                  language="python"
                  theme={isDark ? 'vs-dark' : 'vs'}
                  value={output || (isPlayground ? '# Compiled PyTorch code will appear here...' : '')}
                  loading={<div style={{ padding: '1rem', fontFamily: MONO_FONT, fontSize: '12px', color: 'var(--ifm-color-emphasis-500)' }}>Loading editor...</div>}
                  options={{
                    ...sharedEditorOptions,
                    readOnly: true,
                    lineNumbers: isPlayground ? 'on' : 'off',
                    domReadOnly: true,
                  }}
                />
              )}
            </div>
          )}

          {feat.analysis && analysisData && !isPlayground && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
              <button
                onClick={() => setShowAnalysisPanel(!showAnalysisPanel)}
                className={`button button--sm ${showAnalysisPanel ? 'button--primary' : 'button--outline button--primary'}`}
                style={{ fontSize: '12px' }}
              >
                {showAnalysisPanel ? 'Hide Analysis' : 'Show Analysis'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Analysis panel (tutorial) */}
      {feat.analysis && showAnalysisPanel && analysisData && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          borderRadius: '8px',
          border: '1px solid var(--ifm-color-emphasis-300)',
        }}>
          <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '1rem' }}>Shape Analysis</h4>

          {analysisData.neurons && analysisData.neurons.filter(n => !n.is_primitive).map((neuron, idx) => (
            <div key={idx} style={{ marginBottom: '1rem' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {neuron.name}
                {neuron.params.length > 0 && (
                  <span style={{ fontWeight: 'normal', color: 'var(--ifm-color-emphasis-600)' }}>
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
                    <span style={{ color: 'var(--ifm-color-emphasis-600)' }}>in{port.name !== 'default' ? ` ${port.name}` : ''}:</span>
                    <code className="shape-input" style={{
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '0.85rem',
                    }}>{port.shape}</code>
                  </React.Fragment>
                ))}
                {neuron.outputs.map((port, i) => (
                  <React.Fragment key={`out-${i}`}>
                    <span style={{ color: 'var(--ifm-color-emphasis-600)' }}>out{port.name !== 'default' ? ` ${port.name}` : ''}:</span>
                    <code className="shape-output" style={{
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '0.85rem',
                    }}>{port.shape}</code>
                  </React.Fragment>
                ))}
              </div>

              {neuron.connections.length > 0 && (
                <div style={{ marginTop: '0.5rem', paddingLeft: '1rem' }}>
                  <span style={{ color: 'var(--ifm-color-emphasis-600)', fontSize: '0.85rem' }}>Connections:</span>
                  <div style={{
                    fontFamily: MONO_FONT,
                    fontSize: '0.8rem',
                    color: 'var(--ifm-color-emphasis-700)',
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
                  <div style={{ fontSize: '0.85rem', color: 'var(--ifm-color-emphasis-600)' }}>In {matchExpr.neuron}:</div>
                  {matchExpr.arms.map((arm, armIdx) => (
                    <div key={armIdx} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      fontSize: '0.85rem',
                      marginLeft: '1rem',
                      opacity: arm.is_reachable ? 1 : 0.5,
                    }}>
                      <code className="shape-match" style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                      }}>
                        {arm.pattern}
                        {arm.guard && <span style={{ color: 'var(--ifm-color-emphasis-600)' }}> where {arm.guard}</span>}
                      </code>
                      {!arm.is_reachable && (
                        <span style={{ color: 'var(--ifm-color-emphasis-500)', fontSize: '0.8rem' }}>(unreachable)</span>
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
