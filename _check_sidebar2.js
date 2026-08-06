const path = require('path');
const fs = require('fs');
const matter = require('gray-matter');

const dir = '前置知识';
const base = '/前置知识';
const fullDir = path.resolve('/home/wahaha/paper', dir);
const files = fs.readdirSync(fullDir).filter(f => f.endsWith('.md') && f !== 'index.md');

const articles = files.map(file => {
  const content = fs.readFileSync(path.join(fullDir, file), 'utf-8');
  const { data } = matter(content);
  const slug = file.replace(/\.md$/, '');
  return {
    title: data.title || slug.replace(/^\d+[a-z]?_/, '').replace(/_/g, ' '),
    link: base + '/' + slug,
    order: data.order != null ? data.order : 999,
    category: data.category || '未分类',
  };
}).sort((a, b) => a.order - b.order || a.link.localeCompare(b.link));

// Group by category
const groupMap = new Map();
for (const article of articles) {
  const cat = article.category;
  if (!groupMap.has(cat)) groupMap.set(cat, []);
  groupMap.get(cat).push(article);
}

const groups = Array.from(groupMap.entries()).map(([category, items]) => ({
  text: category,
  collapsed: false,
  items: items.map(a => ({ text: a.title, link: a.link }))
}));

console.log('Sidebar groups for /前置知识/:');
for (const g of groups) {
  console.log(`  Group "${g.text}": ${g.items.length} items`);
  g.items.slice(0, 3).forEach(i => console.log(`    - ${i.text}`));
  if (g.items.length > 3) console.log(`    ... (${g.items.length - 3} more)`);
}
console.log('\nTotal items:', articles.length);
