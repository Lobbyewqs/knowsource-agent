const libraries = document.querySelector('#libraries');
const title = document.querySelector('#library-title');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const question = document.querySelector('#question');
const send = document.querySelector('#send');
const hint = document.querySelector('#hint');
const conversations = new Map();
let current = null;
let busy = false;

function state() { return conversations.get(current); }
function render() {
  title.textContent = current || '暂无知识库';
  document.querySelectorAll('.library').forEach(button => {
    button.classList.toggle('active', button.dataset.name === current);
    button.setAttribute('aria-pressed', button.dataset.name === current);
  });
  messages.replaceChildren();
  const entries = current ? state().entries : [];
  welcome.hidden = entries.length > 0 || !current;
  entries.forEach(entry => {
    const row = document.createElement('article'); row.className = `message ${entry.role}`;
    const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = entry.role === 'user' ? '你' : '知';
    const content = document.createElement('div'); content.className = 'content';
    const label = document.createElement('strong'); label.textContent = entry.role === 'user' ? '你' : '知源';
    const body = document.createElement('div'); body.textContent = entry.text;
    content.append(label, body);
    if (entry.sources?.length) {
      const sources = document.createElement('div'); sources.className = 'sources';
      entry.sources.forEach((source, index) => {
        const details = document.createElement('details'); details.className = 'source';
        const summary = document.createElement('summary'); summary.textContent = `[${index + 1}] ${source.source}${source.source.toLowerCase().endsWith('.pdf') ? ` · 第 ${source.page} 页` : ''}`;
        const excerpt = document.createElement('p'); excerpt.textContent = source.text;
        details.append(summary, excerpt); sources.append(details);
      });
      content.append(sources);
    }
    row.append(avatar, content); messages.append(row);
  });
  document.querySelector('#conversation').scrollTop = document.querySelector('#conversation').scrollHeight;
  send.disabled = busy || !current;
}

async function init() {
  try {
    const response = await fetch('/api/libraries');
    if (!response.ok) throw new Error();
    const {libraries: items} = await response.json();
    items.forEach(item => {
      conversations.set(item.name, {entries: [], history: []});
      const button = document.createElement('button'); button.className = 'library'; button.dataset.name = item.name;
      const icon = document.createElement('span'); icon.className = 'icon'; icon.textContent = '▤';
      const name = document.createElement('span'); name.textContent = item.name;
      const status = document.createElement('small'); status.textContent = item.indexed ? '已索引' : '待索引';
      button.append(icon, name, status); button.onclick = () => { current = item.name; render(); };
      libraries.append(button);
    });
    current = items[0]?.name || null;
    if (!current) hint.textContent = '请先在 knowledge_bases 目录添加知识库。';
    render();
  } catch { title.textContent = '无法连接本地服务'; hint.textContent = '请检查服务是否启动，然后刷新页面。'; send.disabled = true; }
}

document.querySelector('#clear').onclick = () => { if (!current || busy) return; conversations.set(current, {entries: [], history: []}); render(); question.focus(); };
document.querySelectorAll('[data-question]').forEach(button => button.onclick = () => { question.value = button.dataset.question; question.focus(); });
question.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); document.querySelector('#chat-form').requestSubmit(); } });
document.querySelector('#chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = question.value.trim(); if (!text || !current || busy) return;
  const kb = current, conversation = state();
  conversation.entries.push({role:'user', text}); question.value = ''; busy = true; hint.textContent = '正在检索并生成回答，首次使用可能需要下载模型…'; render();
  try {
    const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({kb, question:text, history:conversation.history})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '请求失败');
    conversation.entries.push({role:'assistant', text:result.answer, sources:result.sources});
    conversation.history.push(`用户：${text}\n助手：${result.answer}`); conversation.history = conversation.history.slice(-3);
  } catch (error) { conversation.entries.push({role:'assistant', text:`未能完成请求：${error.message}`}); }
  finally { busy = false; hint.textContent = '回答基于当前知识库内容'; render(); }
});
init();
