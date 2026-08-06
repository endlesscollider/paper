const path = require('path');
const fs = require('fs');

const allKeys = new Set();
allKeys.add('/前置知识/');

const dirs = ['论文综述', '工程实践', '工程项目', '每日快报'];
for (const dir of dirs) {
  const fullDir = path.resolve('/home/wahaha/paper', dir);
  if (!fs.existsSync(fullDir)) continue;
  const base = '/' + dir;
  const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md');
  for (const file of files) {
    const slug = file.replace(/\.md$/, '');
    allKeys.add(base + '/' + slug);
  }
  allKeys.add(base + '/');
}

const seriesRoot = path.resolve('/home/wahaha/paper', '系列');
if (fs.existsSync(seriesRoot)) {
  const subdirs = fs.readdirSync(seriesRoot).filter(f => {
    return fs.statSync(path.join(seriesRoot, f)).isDirectory();
  });
  for (const sub of subdirs) {
    allKeys.add('/系列/' + sub + '/');
  }
}
allKeys.add('/transformer_vla_tutorial/');

console.log('Total sidebar keys:', allKeys.size);

// Collect all pages
function collectPages(dir, prefix) {
  const pages = [];
  if (!fs.existsSync(dir)) return pages;
  const items = fs.readdirSync(dir);
  for (const item of items) {
    const full = path.join(dir, item);
    if (item.startsWith('.') || item === 'node_modules') continue;
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      pages.push(...collectPages(full, prefix + item + '/'));
    } else if (item.endsWith('.md')) {
      pages.push(prefix + item);
    }
  }
  return pages;
}

const allPages = collectPages('/home/wahaha/paper', '/').filter(p =>
  !p.startsWith('/.') && !p.startsWith('/node_modules') && !p.startsWith('/40cd')
);

const sortedKeys = [...allKeys].sort((a, b) => b.split('/').length - a.split('/').length);

const unmatched = [];
for (const page of allPages) {
  const matched = sortedKeys.find(k => page.startsWith(k));
  if (!matched) {
    unmatched.push(page);
  }
}

console.log('\nUnmatched pages (no sidebar will show):');
unmatched.forEach(p => console.log('  ', p));
