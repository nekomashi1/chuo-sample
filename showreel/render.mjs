// Render showreel.html to MP4 (H.264 + AAC) with headless Chromium.
//   node render.mjs                     full 900-frame render -> showreel.mp4
//   node render.mjs --stills 1.2,3.1    PNG stills into ./stills (for checking)
// Env: FFMPEG (path to an ffmpeg with libx264), WORKERS (parallel pages), SAMPLES (motion-blur samples)
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }

const here = dirname(fileURLToPath(import.meta.url));
const url = pathToFileURL(join(here, 'showreel.html')).href + '?render=1';
const FPS = 60, FRAMES = 900;
const WORKERS = +(process.env.WORKERS || 4);
const SAMPLES = +(process.env.SAMPLES || 10);
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

const args = process.argv.slice(2);
const browser = await chromium.launch({ args: ['--allow-file-access-from-files', '--disable-web-security'] });
async function page() {
  const p = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('pageerror', e => console.error('pageerror:', e.message));
  p.on('console', m => { if (m.type() === 'error') console.error('console:', m.text()); });
  await p.goto(url);
  await p.evaluate(() => window.READY);
  return p;
}
const png = d => Buffer.from(d.slice(d.indexOf(',') + 1), 'base64');

if (args[0] === '--stills') {
  const times = args[1].split(',').map(Number);
  const dir = join(here, 'stills'); mkdirSync(dir, { recursive: true });
  const p = await page();
  for (const t of times) {
    const d = await p.evaluate(([t, n]) => { window.stillAt(t, n); return document.getElementById('c').toDataURL('image/png'); }, [t, SAMPLES]);
    writeFileSync(join(dir, `t${t.toFixed(3)}.png`), png(d));
    console.log('still', t);
  }
  await browser.close();
  process.exit(0);
}

const audio = join(here, 'showreel-audio.wav');
const out = join(here, 'showreel.mp4');
const ff = spawn(FFMPEG, [
  '-y', '-hide_banner', '-loglevel', 'error',
  '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'png', '-i', '-',
  ...(existsSync(audio) ? ['-i', audio] : []),
  '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-profile:v', 'high', '-tune', 'animation',
  '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
  ...(existsSync(audio) ? ['-c:a', 'aac', '-b:a', '256k', '-shortest'] : []),
  '-movflags', '+faststart', out
], { stdio: ['pipe', 'inherit', 'inherit'] });

const pages = await Promise.all(Array.from({ length: WORKERS }, page));
const done = new Map(); let next = 0, written = 0;
const t0 = Date.now();
async function flush() {
  while (done.has(written)) {
    const buf = done.get(written); done.delete(written);
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    written++;
    if (written % 60 === 0) console.log(`frame ${written}/${FRAMES}  ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }
}
let flushing = Promise.resolve();
await Promise.all(pages.map(async p => {
  while (next < FRAMES) {
    const i = next++;
    const d = await p.evaluate(([i, n]) => window.renderAt(i, n), [i, SAMPLES]);
    done.set(i, png(d));
    flushing = flushing.then(flush);
    // keep memory bounded if one worker races ahead
    while (done.size > WORKERS * 8) await new Promise(r => setTimeout(r, 20));
  }
}));
await flushing;
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close();
console.log('wrote', out, `${((Date.now() - t0) / 1000).toFixed(1)}s`);
