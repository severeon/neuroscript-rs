/**
 * NeuroScript Monarch tokenizer for Monaco Editor.
 *
 * Derived from the PEG grammar in src/grammar/neuroscript.pest.
 * Provides language definition plus light/dark themes.
 */

export const neuroscriptLanguage = {
  defaultToken: '',
  tokenPostfix: '.neuroscript',

  keywords: [
    'neuron', 'use', 'in', 'out', 'impl', 'graph', 'context',
    'match', 'where', 'if', 'elif', 'else', 'unroll', 'external',
    'and', 'or', 'Freeze',
  ],

  portKeywords: ['in', 'out'],

  sectionKeywords: ['graph', 'context', 'impl'],

  controlKeywords: ['match', 'where', 'if', 'elif', 'else', 'unroll'],

  booleans: ['true', 'false'],

  annotations: ['lazy', 'static', 'global'],

  operators: [
    '->', '==', '!=', '<=', '>=', '<', '>', '=',
    '+', '-', '*', '/',
  ],

  tokenizer: {
    root: [
      // Doc comments (/// ...)
      [/\/\/\/.*$/, 'comment.doc'],

      // Line comments (# ...)
      [/#.*$/, 'comment'],

      // Backtick strings
      [/`[^`]*`/, 'string'],

      // Annotations: @lazy, @static, @global
      [/@(lazy|static|global)\b/, 'annotation'],

      // neuron Name — three-group pattern (all chars must be in consecutive groups)
      [/\b(neuron)(\s+)([A-Z][A-Za-z0-9_]*)/, ['keyword.declaration', 'white', 'type.identifier']],

      // Keywords (exact boundary match)
      [/\b(neuron)\b/, 'keyword.declaration'],
      [/\b(use)\b/, 'keyword.import'],
      [/\b(in|out)\b/, 'keyword.port'],
      [/\b(graph|context|impl)\b/, 'keyword.section'],
      [/\b(match|where)\b/, 'keyword.control'],
      [/\b(if|elif|else)\b/, 'keyword.conditional'],
      [/\b(unroll)\b/, 'keyword.loop'],
      [/\b(and|or)\b/, 'keyword.logical'],
      [/\b(external)\b/, 'keyword'],
      [/\b(Freeze)\b/, 'keyword.freeze'],

      // Booleans
      [/\b(true|false)\b/, 'constant.boolean'],

      // PascalCase + ( → function/neuron call
      [/\b([A-Z][A-Za-z0-9_]*)\s*(?=\()/, 'function'],

      // PascalCase bare → type reference
      [/\b[A-Z][A-Za-z0-9_]*\b/, 'type.identifier'],

      // Floats (before integers — more specific match first)
      [/-?\d+\.\d+(?:[eE][+-]?\d+)?/, 'number.float'],

      // Integers
      [/-?\b\d+\b/, 'number'],

      // Arrow operator (before minus)
      [/->/, 'keyword.arrow'],

      // Comparison operators
      [/==|!=|<=|>=|<|>/, 'operator.comparison'],

      // Assignment
      [/(?<![=!<>])=(?!=)/, 'operator.assignment'],

      // Arithmetic operators
      [/[+\-*/]/, 'operator.arithmetic'],

      // Shape bracket → enter shape state
      [/\[/, { token: 'delimiter.bracket.shape', next: '@shape' }],

      // Parentheses, brackets
      [/[()]/, 'delimiter.parenthesis'],

      // Colon, comma, dot
      [/:/, 'delimiter'],
      [/,/, 'delimiter'],
      [/\./, 'delimiter.dot'],

      // snake_case identifiers
      [/\b[a-z_][a-zA-Z0-9_]*\b/, 'variable'],

      // Whitespace
      [/\s+/, 'white'],
    ],

    shape: [
      // Exit shape on ]
      [/\]/, { token: 'delimiter.bracket.shape', next: '@pop' }],

      // Variadic: *name
      [/(\*)([a-z_][a-zA-Z0-9_]*)/, ['wildcard', 'variable']],

      // Wildcard: *
      [/\*/, 'wildcard'],

      // Integers inside shapes
      [/\b\d+\b/, 'number'],

      // Arithmetic operators in dim expressions
      [/[+\-*/]/, 'operator.arithmetic'],

      // Dimension variable names
      [/\b[a-z_][a-zA-Z0-9_]*\b/, 'variable'],

      // Comma separator
      [/,/, 'delimiter'],

      // Parentheses for grouped dim expressions
      [/[()]/, 'delimiter.parenthesis'],

      // Whitespace
      [/\s+/, 'white'],
    ],
  },
};

/**
 * Light theme — based on GitHub light palette, complementing Docusaurus github prism theme.
 */
export const neuroscriptLightTheme = {
  base: 'vs',
  inherit: true,
  rules: [
    { token: 'comment',              foreground: '6a737d', fontStyle: 'italic' },
    { token: 'comment.doc',          foreground: '6a737d', fontStyle: 'italic' },
    { token: 'string',               foreground: '032f62' },
    { token: 'string.impl-ref',      foreground: '032f62' },
    { token: 'annotation',           foreground: 'e36209' },
    { token: 'keyword',              foreground: 'd73a49' },
    { token: 'keyword.declaration',  foreground: 'd73a49', fontStyle: 'bold' },
    { token: 'keyword.import',       foreground: 'd73a49' },
    { token: 'keyword.port',         foreground: 'd73a49' },
    { token: 'keyword.section',      foreground: 'd73a49' },
    { token: 'keyword.control',      foreground: 'd73a49' },
    { token: 'keyword.conditional',  foreground: 'd73a49' },
    { token: 'keyword.loop',         foreground: 'd73a49' },
    { token: 'keyword.logical',      foreground: 'd73a49' },
    { token: 'keyword.freeze',       foreground: 'd73a49' },
    { token: 'keyword.arrow',        foreground: 'd73a49' },
    { token: 'constant.boolean',     foreground: '005cc5' },
    { token: 'number',               foreground: '005cc5' },
    { token: 'number.float',         foreground: '005cc5' },
    { token: 'function',             foreground: '6f42c1' },
    { token: 'type.identifier',      foreground: '6f42c1' },
    { token: 'variable',             foreground: '24292e' },
    { token: 'namespace',            foreground: 'e36209' },
    { token: 'wildcard',             foreground: 'e36209', fontStyle: 'bold' },
    { token: 'operator.comparison',  foreground: 'd73a49' },
    { token: 'operator.assignment',  foreground: 'd73a49' },
    { token: 'operator.arithmetic',  foreground: 'd73a49' },
    { token: 'delimiter',            foreground: '24292e' },
    { token: 'delimiter.bracket.shape', foreground: 'e36209' },
    { token: 'delimiter.parenthesis', foreground: '24292e' },
    { token: 'delimiter.dot',        foreground: '24292e' },
  ],
  colors: {
    'editor.background': '#ffffff',
    'editor.foreground': '#24292e',
  },
};

/**
 * Dark theme — based on Dracula/GitHub dark palette, complementing Docusaurus dracula prism theme.
 */
export const neuroscriptDarkTheme = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment',              foreground: '6272a4', fontStyle: 'italic' },
    { token: 'comment.doc',          foreground: '6272a4', fontStyle: 'italic' },
    { token: 'string',               foreground: 'f1fa8c' },
    { token: 'string.impl-ref',      foreground: 'f1fa8c' },
    { token: 'annotation',           foreground: 'ffb86c' },
    { token: 'keyword',              foreground: 'ff79c6' },
    { token: 'keyword.declaration',  foreground: 'ff79c6', fontStyle: 'bold' },
    { token: 'keyword.import',       foreground: 'ff79c6' },
    { token: 'keyword.port',         foreground: 'ff79c6' },
    { token: 'keyword.section',      foreground: 'ff79c6' },
    { token: 'keyword.control',      foreground: 'ff79c6' },
    { token: 'keyword.conditional',  foreground: 'ff79c6' },
    { token: 'keyword.loop',         foreground: 'ff79c6' },
    { token: 'keyword.logical',      foreground: 'ff79c6' },
    { token: 'keyword.freeze',       foreground: 'ff79c6' },
    { token: 'keyword.arrow',        foreground: 'ff79c6' },
    { token: 'constant.boolean',     foreground: 'bd93f9' },
    { token: 'number',               foreground: 'bd93f9' },
    { token: 'number.float',         foreground: 'bd93f9' },
    { token: 'function',             foreground: '50fa7b' },
    { token: 'type.identifier',      foreground: '8be9fd' },
    { token: 'variable',             foreground: 'f8f8f2' },
    { token: 'namespace',            foreground: 'ffb86c' },
    { token: 'wildcard',             foreground: 'ffb86c', fontStyle: 'bold' },
    { token: 'operator.comparison',  foreground: 'ff79c6' },
    { token: 'operator.assignment',  foreground: 'ff79c6' },
    { token: 'operator.arithmetic',  foreground: 'ff79c6' },
    { token: 'delimiter',            foreground: 'f8f8f2' },
    { token: 'delimiter.bracket.shape', foreground: 'ffb86c' },
    { token: 'delimiter.parenthesis', foreground: 'f8f8f2' },
    { token: 'delimiter.dot',        foreground: 'f8f8f2' },
  ],
  colors: {
    'editor.background': '#282a36',
    'editor.foreground': '#f8f8f2',
  },
};
