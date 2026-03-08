// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// docusaurus-llms-generator is an optional local dependency.
// If not installed, the site builds without llms.txt generation.
// To enable: clone https://github.com/<org>/docusaurus-llms-generator
// next to this repo and run `npm install` (the optionalDependency will resolve).
let llmsGeneratorAvailable = false;
try {
  await import('docusaurus-llms-generator');
  llmsGeneratorAvailable = true;
} catch {
  // Plugin not installed — skip silently
}

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'NeuroScript',
  tagline: 'Neural Architecture Composition Language',
  favicon: 'img/logo.svg',

  // Set the production url of your site here
  url: 'https://neuroscript-lang.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'neuroscript', // Usually your GitHub org/user name.
  projectName: 'neuroscript', // Usually your repo name.

  onBrokenLinks: 'warn',  // Changed to warn for now
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  
  plugins: [
    ...(llmsGeneratorAvailable
      ? [[
          "docusaurus-llms-generator",
          {
            outputFileName: "llms.txt",
            outputFileNameFull: "llms-full.txt",
          },
        ]]
      : []),
    require.resolve('./plugins/neuron-docs-plugin'),
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/neuroscript/neuroscript/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themes: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      ({
        hashed: true,
        language: ["en"],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      hashed: true,
      language: ["en"],
      highlightSearchTermsOnTargetPage: true,
      explicitSearchResultPath: true,
      navbar: {
        title: 'NeuroScript',
        logo: {
          alt: 'NeuroScript Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            to: '/playground',
            label: 'Playground',
            position: 'left',
          },
          {
            label: 'Neuron Genealogy',
            to: '/neuron-genealogy',
            position: 'left',
          },
          {
            href: 'https://github.com/neuroscript/neuroscript',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
              {
                label: 'Primitives',
                to: '/docs/primitives',
              },
              {
                label: 'Standard Library',
                to: '/docs/stdlib',
              },
              {
                label: 'Packages',
                to: '/docs/packages',
              },
              {
                label: 'Playground',
                to: '/playground',
              },
              {
                label: 'Neuron Genealogy',
                to: '/neuron-genealogy',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/neuroscript/neuroscript',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} NeuroScript Project. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['rust', 'python'],
      },
    }),
};

export default config;
