const fs = require('fs');
const path = require('path');

module.exports = function neuronDocsPlugin(context) {
  return {
    name: 'neuron-docs-plugin',

    async contentLoaded({ actions }) {
      const { createData } = actions;
      const stdlibDir = path.resolve(context.siteDir, '..', 'stdlib');

      let files;
      try {
        files = fs.readdirSync(stdlibDir).filter(f => f.endsWith('.ns'));
      } catch {
        files = [];
      }

      const docs = {};
      for (const file of files) {
        const filePath = path.join(stdlibDir, file);
        const source = fs.readFileSync(filePath, 'utf-8');
        const lines = source.split('\n');

        // Extract all neuron definitions and their preceding /// doc comments
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].trimStart().startsWith('neuron ')) {
            const docLines = [];
            let j = i - 1;
            while (j >= 0 && lines[j].trimStart().startsWith('///')) {
              docLines.unshift(lines[j].trimStart().replace(/^\/\/\/\s?/, ''));
              j--;
            }
            const match = lines[i].match(/neuron\s+([A-Za-z0-9_]+)/);
            if (match) {
              docs[match[1]] = {
                docComment: docLines.join('\n'),
                source: source,
                sourceFile: `stdlib/${file}`,
              };
            }
          }
        }
      }

      await createData('neuron-docs.json', JSON.stringify(docs));
    },
  };
};
