const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const outDir = process.argv[2];
  const base = 'http://127.0.0.1:8502';
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1720, height: 1100 } });
  const targets = [
    { name: 'root', url: base },
    { name: 'trader', url: base + '/Trader' },
    { name: 'controle', url: base + '/Controle_do_Bot' },
  ];
  const info = {};
  for (const t of targets) {
    try {
      await page.goto(t.url, { waitUntil: 'domcontentloaded', timeout: 45000 });
      await page.waitForTimeout(7000);
      const screenshotPath = path.join(outDir, `${t.name}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      const text = await page.locator('body').innerText();
      info[t.name] = {
        url: t.url,
        screenshot: screenshotPath,
        has_trava: /Trava|perda di?ria|perda diaria|Limite di?rio|Limite diario/i.test(text),
        has_trader: /Trader Premium Max|Trader/i.test(text),
        has_controle: /Controle Operacional|Controle do Bot/i.test(text),
      };
    } catch (err) {
      info[t.name] = { url: t.url, error: String(err) };
    }
  }
  fs.writeFileSync(path.join(outDir, 'screenshots_info.json'), JSON.stringify(info, null, 2), 'utf-8');
  await browser.close();
})();