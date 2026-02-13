/**
 * Register NeuroScript language, Monarch tokenizer, and themes with Monaco.
 *
 * Call `registerNeuroScript(monaco)` once in the Editor's `beforeMount` callback.
 *
 * Note: monaco-editor is loaded from CDN (jsDelivr) by @monaco-editor/loader,
 * which is a dependency of @monaco-editor/react. No local monaco-editor package
 * is needed in package.json — this is the intended setup per the library docs.
 */
import {
  neuroscriptLanguage,
  neuroscriptLightTheme,
  neuroscriptDarkTheme,
} from './neuroscript-monarch';

export function registerNeuroScript(monaco) {
  // Avoid double-registration
  const registered = monaco.languages.getLanguages().some(l => l.id === 'neuroscript');
  if (registered) return;

  monaco.languages.register({
    id: 'neuroscript',
    extensions: ['.ns'],
    aliases: ['NeuroScript', 'neuroscript'],
  });

  monaco.languages.setMonarchTokensProvider('neuroscript', neuroscriptLanguage);

  monaco.languages.setLanguageConfiguration('neuroscript', {
    comments: {
      lineComment: '#',
    },
    brackets: [
      ['(', ')'],
      ['[', ']'],
    ],
    autoClosingPairs: [
      { open: '(', close: ')' },
      { open: '[', close: ']' },
      { open: '`', close: '`' },
    ],
    surroundingPairs: [
      { open: '(', close: ')' },
      { open: '[', close: ']' },
      { open: '`', close: '`' },
    ],
    indentationRules: {
      increaseIndentPattern: /^\s*(neuron|graph|context|match|if|elif|else|unroll)\b.*:\s*$/,
      decreaseIndentPattern: /^\s*$/,
    },
    wordPattern: /[a-zA-Z_][a-zA-Z0-9_]*/,
  });

  monaco.editor.defineTheme('neuroscript-light', neuroscriptLightTheme);
  monaco.editor.defineTheme('neuroscript-dark', neuroscriptDarkTheme);
}
