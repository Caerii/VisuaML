#!/usr/bin/env node
/**
 * Python Environment Checker for VisuaML
 * 
 * This script helps developers verify their Python setup is correctly configured.
 */

import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });
dotenv.config();

const PYTHON_COMMAND = process.env.VISUAML_PYTHON || 'python';

console.log('🐍 VisuaML Python Environment Checker\n');

// Check Python version
try {
  console.log('1. Checking Python version...');
  const version = execSync(`${PYTHON_COMMAND} --version`, { encoding: 'utf8' }).trim();
  console.log(`   ✅ Found: ${version}`);
  
  const versionMatch = version.match(/Python (\d+)\.(\d+)\.(\d+)/);
  if (versionMatch) {
    const [, major, minor] = versionMatch.map(Number);
    if (major < 3 || (major === 3 && minor < 11)) {
      console.log('   ❌ Error: Python 3.11+ required for VisuaML');
      console.log('   📖 See visuaml-client/PYTHON_SETUP.md for setup instructions');
      process.exit(1);
    }
  }
} catch (error) {
  console.log(`   ❌ Error: Could not run '${PYTHON_COMMAND}'`);
  console.log('   💡 Tip: Set VISUAML_PYTHON environment variable to your Python path');
  console.log('   📖 See visuaml-client/PYTHON_SETUP.md for setup instructions');
  process.exit(1);
}

// Check if we can import required packages
try {
  console.log('\n2. Checking Python packages...');
  const packages = ['torch', 'open_hypergraphs', 'numpy'];
  
  for (const pkg of packages) {
    try {
      execSync(`${PYTHON_COMMAND} -c "import ${pkg}; print('${pkg}: OK')"`, { 
        encoding: 'utf8',
        stdio: 'pipe',
        env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE' }
      });
      console.log(`   ✅ ${pkg}: Available`);
    } catch (error) {
      console.log(`   ❌ ${pkg}: Missing or broken`);
      throw new Error(`Package ${pkg} not available`);
    }
  }
} catch (error) {
  console.log('\n   💡 Install packages with:');
  console.log(`   ${PYTHON_COMMAND} -m pip install -r backend/requirements.txt`);
  console.log('   📖 See visuaml-client/PYTHON_SETUP.md for detailed instructions');
  process.exit(1);
}

// Check if fx_export.py works
try {
  console.log('\n3. Testing backend script...');
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const scriptPath = path.resolve(__dirname, '..', 'backend', 'scripts', 'fx_export.py');
  execSync(`${PYTHON_COMMAND} "${scriptPath}" --help`, { 
    stdio: 'pipe',
    env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE' }
  });
  console.log('   ✅ Backend script: Working');
} catch (error) {
  console.log('   ❌ Backend script: Failed');
  console.log('   📖 See visuaml-client/PYTHON_SETUP.md for troubleshooting');
  process.exit(1);
}

console.log('\n🎉 Python environment is properly configured!');
console.log('\n📝 Current configuration:');
console.log(`   VISUAML_PYTHON = ${PYTHON_COMMAND}`);
if (process.env.VISUAML_PYTHON) {
  console.log('   (Set via environment variable)');
} else {
  console.log('   (Using system default)');
} 