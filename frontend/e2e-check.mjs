// End-to-end check of the Village Pond Planner UI.
// Drives the real browser: search a village, draw an area, analyse it,
// then apply a budget, capturing screenshots and any console errors.
import { chromium } from 'playwright';

const BASE = 'http://127.0.0.1:5173';
const OUT = process.argv[2] || '/tmp/pond_shots';

const log = (...a) => console.log(...a);

const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });

const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text().slice(0, 300));
});
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + String(e).slice(0, 300)));

await page.goto(BASE, { waitUntil: 'networkidle' });
log('1. loaded:', await page.title());
await page.screenshot({ path: `${OUT}/01-initial.png` });

// ── Village search ──────────────────────────────────────────────────────────
await page.fill('.search-input', 'Jeora');
await page.waitForSelector('.search-result', { timeout: 15000 });
const options = await page.$$eval('.search-result', (els) =>
  els.map((e) => e.innerText.replace(/\n/g, ' | ')),
);
log('2. suggestions:', options.length);
options.slice(0, 3).forEach((o) => log('     ', o));

// Pick the Durg, Chhattisgarh one (matches the sample terrain)
const durgIndex = options.findIndex((o) => /Durg/.test(o));
await page.$$eval(
  '.search-result',
  (els, i) => els[i].dispatchEvent(new MouseEvent('mousedown', { bubbles: true })),
  durgIndex >= 0 ? durgIndex : 0,
);
await page.waitForTimeout(4000);
log('3. selected village:', (await page.$eval('.village-chip', (e) => e.innerText.replace(/\n/g, ' | ')).catch(() => 'none')));
await page.screenshot({ path: `${OUT}/02-village.png` });

// ── Draw a rectangle ────────────────────────────────────────────────────────
const rectButton = await page.$('.leaflet-pm-icon-rectangle');
if (!rectButton) throw new Error('Geoman rectangle tool did not render');
await rectButton.click();
await page.waitForTimeout(500);

const map = await page.$('.leaflet-container');
const box = await map.boundingBox();
const cx = box.x + box.width / 2;
const cy = box.y + box.height / 2;
await page.mouse.move(cx - 130, cy - 90);
await page.mouse.click(cx - 130, cy - 90);
await page.waitForTimeout(300);
await page.mouse.move(cx + 130, cy + 90, { steps: 12 });
await page.mouse.click(cx + 130, cy + 90);
await page.waitForTimeout(1500);

const selectionText = await page.$eval('.selection-state', (e) => e.innerText.trim());
log('4. selection:', selectionText);
await page.screenshot({ path: `${OUT}/03-drawn.png` });

// ── Analyse ─────────────────────────────────────────────────────────────────
await page.click('button:has-text("Analyse this area")');
log('5. analysing…');
await page.waitForSelector('.site-card, .error-banner, .panel-empty', { timeout: 180000 });
await page.waitForTimeout(2500);

const err = await page.$('.error-banner');
if (err) log('   ERROR BANNER:', (await err.innerText()).slice(0, 200));

const cards = await page.$$('.site-card');
log('6. site cards:', cards.length);
for (const card of cards.slice(0, 4)) {
  log('     ', (await card.innerText()).replace(/\n+/g, ' | ').slice(0, 170));
}
const overview = await page.$eval('.stat-grid', (e) => e.innerText.replace(/\n+/g, ' | ')).catch(() => 'n/a');
log('7. overview:', overview);
await page.screenshot({ path: `${OUT}/04-results.png` });

// Overlays actually drawn on the map?
const overlays = await page.evaluate(() => ({
  paths: document.querySelectorAll('.leaflet-overlay-pane path').length,
  markers: document.querySelectorAll('.leaflet-overlay-pane circle, .leaflet-marker-icon').length,
  legend: !!document.querySelector('.map-legend'),
}));
log('8. overlays:', JSON.stringify(overlays));

// ── Budget filter ───────────────────────────────────────────────────────────
await page.click('button[role="tab"]:has-text("Budget filter")');
await page.waitForSelector('#budget-input', { timeout: 10000 });
await page.fill('#budget-input', '800000');
await page.click('button:has-text("Apply budget")');
await page.waitForSelector('.headline-value, .error-banner', { timeout: 60000 });
await page.waitForTimeout(1500);

const headline = await page.$eval('.headline-value', (e) => e.innerText).catch(() => null);
const headlineLabel = await page.$eval('.headline-label', (e) => e.innerText).catch(() => '');
log('9. budget result:', headline, '—', headlineLabel);
const designRows = await page.$$eval('.budget-result .stat-rows .stat-row', (els) =>
  els.map((e) => e.innerText.replace(/\n/g, ': ')),
);
designRows.forEach((r) => log('     ', r));
const costRows = await page.$$eval('.cost-breakdown .stat-row', (els) =>
  els.map((e) => e.innerText.replace(/\n/g, ': ')),
);
costRows.forEach((r) => log('     ', r));
await page.screenshot({ path: `${OUT}/05-budget.png` });

const after = await page.evaluate(() => document.querySelectorAll('.leaflet-overlay-pane path').length);
log('10. overlay paths after earthwork:', after);

// ── On-map layer switches ───────────────────────────────────────────────────
const panelRows = await page.$$('.map-layer-row');
log('11. layer switches on the map:', panelRows.length);
// Turn on the chosen pond's catchment and check something new is drawn
const before = await page.evaluate(() => document.querySelectorAll('.leaflet-overlay-pane canvas').length);
await page.click('.map-layer-row:has-text("Catchment of chosen pond") input');
await page.waitForTimeout(900);
log('12. site catchment toggled, canvases:', before, '->',
    await page.evaluate(() => document.querySelectorAll('.leaflet-overlay-pane canvas').length));
const pins = await page.$$('.site-pin');
log('13. numbered pins:', pins.length,
    'selected:', await page.$$eval('.site-pin.is-selected', (e) => e.length));
await page.screenshot({ path: `${OUT}/06-layers.png` });

// Zoom past the imagery's native level: this used to show "Map data not yet
// available" tiles instead of upscaling.
for (let i = 0; i < 6; i += 1) {
  const disabled = await page.$eval('.leaflet-control-zoom-in',
    (e) => e.classList.contains('leaflet-disabled')).catch(() => true);
  if (disabled) break;                       // Leaflet greys the button at max zoom
  await page.click('.leaflet-control-zoom-in');
  await page.waitForTimeout(600);
}
await page.waitForTimeout(3000);
const tileState = await page.evaluate(() => {
  const imgs = [...document.querySelectorAll('.leaflet-tile')];
  return {
    total: imgs.length,
    loaded: imgs.filter((i) => i.complete && i.naturalWidth > 0).length,
    broken: imgs.filter((i) => i.complete && i.naturalWidth === 0).length,
  };
});
log('14. zoom now', await page.$eval('.zoom-level', (e) => e.innerText),
    '| tiles', JSON.stringify(tileState));
await page.screenshot({ path: `${OUT}/07-zoomed.png` });

// Only the chosen pond should be drawn, and choosing another should swap it
await page.click('button[role="tab"]:has-text("Results")');
await page.waitForTimeout(700);
const siteCards = await page.$$('.site-card');
log('15. one pond at a time: cards', siteCards.length);
if (siteCards.length > 1) {
  await siteCards[1].click();
  await page.waitForTimeout(1200);
  const rank = await page.$eval('.site-card.active .site-rank', (e) => e.innerText);
  const pinSel = await page.$$eval('.site-pin.is-selected', (e) => e.map((x) => x.innerText));
  const drawn = await page.evaluate(() => document.querySelectorAll('.leaflet-overlay-pane canvas').length);
  log('    picked card', rank, '| highlighted pin', JSON.stringify(pinSel), '| canvases', drawn);
  await page.screenshot({ path: `${OUT}/08-second-pond.png` });
}

log('\nconsole errors:', errors.length);
errors.slice(0, 8).forEach((e) => log('   !', e));

await browser.close();
log('\nDONE');
