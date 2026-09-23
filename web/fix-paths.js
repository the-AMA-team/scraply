#!/usr/bin/env node
/**
 * Fix paths in the exported Next.js files for Electron
 * This script converts absolute paths to relative paths in the HTML files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function fixPaths() {
  const outDir = path.join(__dirname, 'out');

  if (!fs.existsSync(outDir)) {
    console.error('out directory not found');
    return;
  }

  const extsToProcess = new Set(['.html', '.css', '.js']);

  /**
   * @param {string} filePath
   */
  function processFile(filePath) {
    try {
      let content = fs.readFileSync(filePath, 'utf8');
      const original = content;

      // Replace absolute Next asset paths with relative ones
      content = content.replace(/href=(\"|\')\/_next\//g, 'href=$1./_next/');
      content = content.replace(/src=(\"|\')\/_next\//g, 'src=$1./_next/');
      content = content.replace(/url\((\"|\')?\/_next\//g, 'url($1./_next/');
      content = content.replace(/(\"|\')\/_next\//g, '$1./_next/');

      if (content !== original) {
        fs.writeFileSync(filePath, content);
      }
    } catch (err) {
      // Ignore binary or unreadable files
    }
  }

  /**
   * @param {string} dir
   */
  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (extsToProcess.has(path.extname(entry.name))) {
        processFile(fullPath);
      }
    }
  }

  walk(outDir);

  console.log('Fixed Next asset paths to be relative for Electron');
}

fixPaths();
