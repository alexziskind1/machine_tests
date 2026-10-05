import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const port = 9222;
const profile = fileURLToPath(new URL('./chrome-profile', import.meta.url));
const url = 'https://browserbench.org/Speedometer3.1/?startAutomatically=true';
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function connect() {
  try { return await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}`, defaultViewport: null }); }
  catch { return null; }
}

// Close the benchmark Chrome left open by a previous run, then start from a fresh profile.
const old = await connect();
if (old) { await old.close(); await sleep(2000); }
rmSync(profile, { recursive: true, force: true });

// Launch through LaunchServices so Chrome is not a child of this script (or of an SSH session)
// and stays open after the script exits.
execFileSync('open', ['-na', 'Google Chrome', '--args',
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--window-size=1400,1000', 'about:blank']);

let browser = null;
for (let i = 0; i < 30 && !browser; i++) { await sleep(1000); browser = await connect(); }
if (!browser) throw new Error('Could not connect to Chrome');

const page = (await browser.pages())[0] ?? await browser.newPage();
await page.bringToFront();
const t0 = Date.now();
await page.goto(url, { waitUntil: 'load' });
await page.waitForFunction(() => {
  const el = document.querySelector('#result-number');
  return el && el.textContent.trim() && location.hash.includes('summary');
}, { timeout: 30 * 60 * 1000, polling: 2000 });
const r = await page.evaluate(() => ({
  score: document.querySelector('#result-number')?.textContent.trim(),
  confidence: document.querySelector('#confidence-number')?.textContent.trim(),
  ua: navigator.userAgent,
}));
r.seconds = Math.round((Date.now() - t0) / 1000);
console.log(JSON.stringify(r));
// Detach without closing, so the score stays on screen.
await browser.disconnect();
