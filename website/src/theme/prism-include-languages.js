/**
 * Swizzled Docusaurus prism-include-languages to register NeuroScript.
 *
 * This file is picked up automatically by Docusaurus and adds NeuroScript
 * syntax highlighting to all ```neuroscript code blocks in docs.
 */

import siteConfig from '@generated/docusaurus.config';

export default function prismIncludeLanguages(PrismObject) {
  const {
    themeConfig: { prism: { additionalLanguages = [] } = {} },
  } = siteConfig;

  // Let Docusaurus load its configured additional languages first
  globalThis.Prism = PrismObject;
  additionalLanguages.forEach((lang) => {
    require(`prismjs/components/prism-${lang}`);
  });
  delete globalThis.Prism;

  // Register NeuroScript
  PrismObject.languages.neuroscript = {
    comment: [
      {
        pattern: /\/\/\/.*/,
        alias: 'doc-comment',
        greedy: true,
      },
      {
        pattern: /#.*/,
        greedy: true,
      },
    ],

    string: {
      pattern: /`[^`]*`/,
      greedy: true,
    },

    annotation: {
      pattern: /@(?:lazy|static|global)\b/,
      alias: 'builtin',
    },

    'neuron-declaration': {
      pattern: /\b(neuron)\s+([A-Z][A-Za-z0-9_]*)/,
      inside: {
        keyword: /\bneuron\b/,
        'class-name': /[A-Z][A-Za-z0-9_]*/,
      },
    },

    'impl-ref': {
      pattern: /(?<=impl\s*:\s*)[a-z_][a-zA-Z0-9_]*\s*,\s*[a-zA-Z0-9_/*]+(?:\/[a-zA-Z0-9_/*]+)*/,
      alias: 'string',
    },

    boolean: /\b(?:true|false)\b/,

    keyword: /\b(?:neuron|use|in|out|impl|graph|context|match|where|if|elif|else|unroll|external|and|or|Freeze)\b/,

    'function-call': {
      pattern: /\b[A-Z][A-Za-z0-9_]*(?=\s*\()/,
      alias: 'function',
    },

    'class-name': /\b[A-Z][A-Za-z0-9_]*\b/,

    number: [
      {
        pattern: /-?\b\d+\.\d+(?:[eE][+-]?\d+)?\b/,
        alias: 'float',
      },
      {
        pattern: /-?\b\d+\b/,
        alias: 'integer',
      },
    ],

    operator: [
      {
        pattern: /->/,
        alias: 'arrow',
      },
      /==|!=|<=|>=|[<>]/,
      /[+\-*/]/,
      {
        pattern: /(?<![=!<>])=(?!=)/,
        alias: 'assignment',
      },
    ],

    'shape-bracket': {
      pattern: /\[(?:[^\]]*)\]/,
      inside: {
        wildcard: {
          pattern: /\*[a-z_][a-zA-Z0-9_]*|\*/,
          alias: 'builtin',
        },
        number: /-?\b\d+\b/,
        variable: /\b[a-z_][a-zA-Z0-9_]*\b/,
        operator: /[+\-*/]/,
        punctuation: /[,[\]()]/,
      },
    },

    variable: /\b[a-z_][a-zA-Z0-9_]*\b/,

    punctuation: /[()[\]:,.]|->/,
  };
}
