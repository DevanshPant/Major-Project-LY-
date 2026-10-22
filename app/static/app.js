/* ResumeAI frontend logic.
   No framework. Just fetch + DOM.  Contrast/legibility rule:
   - All produced text uses --text or explicit chip colors.
   - Never render text with color:transparent or color:var(--bg). */

let selectedFiles = [];   // File[]
let jdFile = null;        // File | null (uploaded JD)

// ---------- theme toggle ----------
(function initTheme() {
  const btn = () => document.getElementById('themeToggle');
  const apply = (t) => {
    if (t === 'light') document.documentElement.setAttribute('data-theme', 'light');
    else document.documentElement.removeAttribute('data-theme');
  };
  const wire = () => {
    const b = btn();
    if (!b) return;
    b.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
      const next = cur === 'light' ? 'dark' : 'light';
      apply(next);
      try { localStorage.setItem('resumeai-theme', next); } catch (e) {}
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();

function bytesFmt(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

function alertBox(msg, kind = 'error') {
  const el = document.getElementById('alert');
  if (!el) return;
  el.innerHTML = `<div class="alert ${kind === 'info' ? 'info' : ''}">${msg}</div>`;
}
function clearAlert() { const el = document.getElementById('alert'); if (el) el.innerHTML = ''; }

const ACCEPTED_RESUME_EXT = /\.(pdf|docx|txt|zip)$/i;

function wireDropzone() {
  const drop      = document.getElementById('drop');
  const input     = document.getElementById('files');
  const folderIn  = document.getElementById('filesFolder');
  const pickBtn   = document.getElementById('pickFilesBtn');
  const folderBtn = document.getElementById('pickFolderBtn');
  const clearBtn  = document.getElementById('clearFilesBtn');
  const summary   = document.getElementById('fileSummary');
  const list      = document.getElementById('fileList');
  if (!drop) return;

  // Break down what the user selected — count by extension, hide the noise.
  const summarise = () => {
    if (!selectedFiles.length) {
      list.innerHTML = '';
      if (summary) summary.classList.add('hidden');
      clearBtn?.classList.add('hidden');
      return;
    }
    const total = selectedFiles.reduce((a, f) => a + f.size, 0);
    const byExt = {};
    for (const f of selectedFiles) {
      const m = f.name.toLowerCase().match(/\.[a-z0-9]+$/);
      const e = m ? m[0] : '(none)';
      byExt[e] = (byExt[e] || 0) + 1;
    }
    const parts = Object.entries(byExt)
      .sort((a, b) => b[1] - a[1])
      .map(([e, n]) => `<b>${n}</b> ${escapeHtml(e)}`)
      .join(' &middot; ');
    if (summary) {
      summary.classList.remove('hidden');
      summary.innerHTML = `
        <div>
          <b>${selectedFiles.length}</b> file${selectedFiles.length !== 1 ? 's' : ''} selected
          <span class="dim"> &middot; ${bytesFmt(total)} total</span>
        </div>
        <div class="dim" style="margin-top:4px">${parts}</div>`;
    }
    clearBtn?.classList.remove('hidden');

    // Only show the first 15 filenames — anything more is noise for 2000 files.
    list.innerHTML = '';
    selectedFiles.slice(0, 15).forEach(f => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${escapeHtml(f.name)}</span><span class="size">${bytesFmt(f.size)}</span>`;
      list.appendChild(li);
    });
    if (selectedFiles.length > 15) {
      const li = document.createElement('li');
      li.style.background = 'var(--bg-2)';
      li.innerHTML = `<span class="dim">&hellip; ${selectedFiles.length - 15} more file(s) hidden</span>
                      <span class="size">${bytesFmt(total)} total</span>`;
      list.appendChild(li);
    }
  };

  // Filter and append — drops any file whose extension we can't parse.
  const addFiles = (fileList) => {
    const incoming = [...fileList];
    let skipped = 0;
    for (const f of incoming) {
      if (!ACCEPTED_RESUME_EXT.test(f.name)) { skipped++; continue; }
      selectedFiles.push(f);
    }
    if (skipped) {
      alertBox(`Skipped ${skipped} file(s) with unsupported extensions (need .pdf, .docx, .txt or .zip)`, 'info');
      setTimeout(clearAlert, 4000);
    }
    summarise();
  };

  pickBtn?.addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  folderBtn?.addEventListener('click', (e) => { e.stopPropagation(); folderIn.click(); });
  clearBtn?.addEventListener('click', (e) => { e.stopPropagation(); selectedFiles = []; summarise(); });

  // Clicking on the dropzone body (but not on the buttons) picks files too.
  drop.addEventListener('click', (e) => {
    if (e.target === drop || e.target.parentElement === drop) input.click();
  });

  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag');
    addFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', e => {
    addFiles(e.target.files);
    e.target.value = '';
  });
  folderIn?.addEventListener('change', e => {
    addFiles(e.target.files);
    e.target.value = '';
  });
}

function wireJdDropzone() {
  const drop = document.getElementById('jdDrop');
  const input = document.getElementById('jdFile');
  const label = document.getElementById('jdDropLabel');
  if (!drop || !input || !label) return;
  const setFile = (f) => {
    jdFile = f || null;
    label.textContent = jdFile
      ? `JD file: ${jdFile.name} (${bytesFmt(jdFile.size)}) — click to change`
      : 'Drop a JD file';
  };
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('drag');
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', e => {
    if (e.target.files.length) setFile(e.target.files[0]);
    e.target.value = '';
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function qualityBadgeClass(q) {
  return ({ Excellent: 'good', Strong: 'strong', Good: 'strong',
           Moderate: 'moderate', Weak: 'weak' }[q] || '');
}

function renderMatch(m) {
  const chips = (arr, cls = '') =>
    arr.length
      ? arr.slice(0, 15).map(s => `<span class="chip ${cls}">${escapeHtml(s)}</span>`).join('')
      : `<span class="chip ${cls}" style="opacity:.6">none</span>`;

  return `
    <div class="result-card">
      <div class="result-top">
        <div class="rank">${m.rank ?? '·'}</div>
        <div class="file">${escapeHtml(m.file)}</div>
        <div class="score">${m.final_score}<small>/100</small></div>
      </div>
      <div style="margin-bottom:10px; display:flex; gap:8px; flex-wrap:wrap;">
        <span class="badge ${qualityBadgeClass(m.quality)}">${m.quality}</span>
        ${m.suspicious ? `<span class="badge weak" title="${escapeHtml((m.flags||[]).join(' · '))}">⚠ Suspicious resume</span>` : ''}
      </div>
      ${m.suspicious ? `
        <div class="alert" style="margin:0 0 12px;">
          <b>Integrity flags:</b>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${(m.flags||[]).map(f => `<li>${escapeHtml(f)}</li>`).join('')}
          </ul>
          ${m.hidden_text ? `<div style="margin-top:8px; font-size:12px; color:var(--muted)">Hidden/invisible text extracted: <code style="color:var(--ink)">${escapeHtml(m.hidden_text)}</code></div>` : ''}
        </div>` : ''}
      <div class="bars">
        ${bar('Semantic', m.semantic)}
        ${bar('Skills',   m.skills)}
        ${bar('Experience', m.experience)}
        ${bar('Education',  m.education)}
      </div>
      <div class="meta-line">
        <span>Experience: <b>${m.years} yr</b></span>
        <span>Degree: <b>${escapeHtml(m.degree)}</b></span>
      </div>
      ${m.explanation ? `
        <div class="reason">
          <div class="reason-headline">${escapeHtml(m.explanation.headline)}</div>
          ${m.explanation.strengths.length ? `
            <div class="reason-block">
              <div class="reason-label good">Why chosen</div>
              <ul>${m.explanation.strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
            </div>` : ''}
          ${m.explanation.concerns.length ? `
            <div class="reason-block">
              <div class="reason-label warn">Concerns</div>
              <ul>${m.explanation.concerns.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
            </div>` : ''}
        </div>` : ''}
      ${m.matched_skills ? `
        <div style="margin-top:12px">
          <div style="font-size:12px; color:var(--muted); margin-bottom:4px">Matched skills</div>
          <div class="chips">${chips((m.matched_skills || '').split(', ').filter(Boolean))}</div>
        </div>` : ''}
      ${m.missing_skills ? `
        <div style="margin-top:10px">
          <div style="font-size:12px; color:var(--muted); margin-bottom:4px">Missing from JD</div>
          <div class="chips">${chips((m.missing_skills || '').split(', ').filter(Boolean), 'miss')}</div>
        </div>` : ''}
      ${m.snippet ? `<div class="snippet">${escapeHtml(m.snippet)}</div>` : ''}
    </div>`;
}
function bar(label, pct) {
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  return `<div class="bar">
    <div class="lbl"><span>${label}</span><b>${p}</b></div>
    <div class="track"><div class="fill" style="width:${p}%"></div></div>
  </div>`;
}

function setLoader(on, txt) {
  const l = document.getElementById('loader');
  if (!l) return;
  l.classList.toggle('on', !!on);
  const t = document.getElementById('loaderText');
  if (txt && t) t.textContent = txt;
}

async function initAdmin() {
  wireDropzone();
  wireJdDropzone();
  let lastResults = null;
  const btn = document.getElementById('goBtn');
  const dlBtn = document.getElementById('dlBtn');

  btn.addEventListener('click', async () => {
    clearAlert();
    const jd = document.getElementById('jd').value.trim();
    const topN = parseInt(document.getElementById('topn').value || '10', 10);
    const rerank = document.getElementById('rerank').checked;
    const bias = document.getElementById('bias').checked;
    // Integrity check and OCR are handled automatically by the server:
    //   - Integrity always runs on the top-N shortlist (cheap)
    //   - OCR auto-enables when the batch is small (<= 20 files) and
    //     has scanned PDFs; skipped silently on huge batches.
    if (!jd && !jdFile) return alertBox('Provide a JD — paste text or upload a file.');
    if (!selectedFiles.length) return alertBox('Upload at least one resume or a ZIP.');

    const fd = new FormData();
    fd.append('jd', jd);
    fd.append('top_n', String(topN));
    fd.append('use_reranker', rerank ? 'true' : 'false');
    fd.append('bias_check', bias ? 'true' : 'false');
    if (jdFile) fd.append('jd_file', jdFile, jdFile.name);
    selectedFiles.forEach(f => fd.append('files', f, f.name));

    btn.disabled = true;
    const nFiles = selectedFiles.length;
    setLoader(true,
      nFiles > 100 ? `Scoring ${nFiles} files — this may take a minute…`
      : nFiles > 20 ? `Scoring ${nFiles} files…`
      : `Scoring ${nFiles} file${nFiles > 1 ? 's' : ''}…`);
    document.getElementById('resultsWrap').classList.add('hidden');
    try {
      const r = await fetch('/api/match', { method: 'POST', body: fd });
      if (r.status === 401) { window.location.href = '/login'; return; }
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || 'Matching failed');

      lastResults = j.results;
      const wrap = document.getElementById('resultsWrap');
      wrap.classList.remove('hidden');
      const unreadNote = j.count_unreadable ? ` · ${j.count_unreadable} unreadable skipped` : '';
      const susNote = j.count_suspicious ? ` · ${j.count_suspicious} flagged as suspicious` : '';
      document.getElementById('resultsTitle').textContent =
        `Top ${j.results.length} of ${j.count_uploaded} resumes${unreadNote}${susNote}`;
      if (j.ocr_note) {
        const noteEl = document.createElement('div');
        noteEl.className = 'alert info';
        noteEl.textContent = j.ocr_note;
        document.getElementById('resultsWrap').insertBefore(noteEl, document.getElementById('results'));
      }
      document.getElementById('modelBadge').textContent =
        `reranker: ${j.reranker}`;
      document.getElementById('results').innerHTML =
        j.results.map(renderMatch).join('') ||
        '<div class="alert info">No matches produced.</div>';
      dlBtn.classList.remove('hidden');
    } catch (e) {
      alertBox(e.message);
    } finally {
      btn.disabled = false; setLoader(false);
    }
  });

  dlBtn.addEventListener('click', async () => {
    if (!lastResults) return;
    const fd = new FormData();
    fd.append('payload', JSON.stringify(lastResults));
    const r = await fetch('/api/export', { method: 'POST', body: fd });
    if (!r.ok) return alertBox('Export failed');
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'shortlist.csv';
    document.body.appendChild(a); a.click(); a.remove();
  });
}

async function initCompare() {
  wireDropzone();
  wireJdDropzone();
  const btn = document.getElementById('goBtn');
  btn.addEventListener('click', async () => {
    clearAlert();
    const jd = document.getElementById('jd').value.trim();
    const topN = parseInt(document.getElementById('topn').value || '10', 10);
    if (!jd && !jdFile) return alertBox('Provide a JD — paste text or upload a file.');
    if (!selectedFiles.length) return alertBox('Upload resumes first');

    const fd = new FormData();
    fd.append('jd', jd); fd.append('top_n', String(topN));
    if (jdFile) fd.append('jd_file', jdFile, jdFile.name);
    selectedFiles.forEach(f => fd.append('files', f, f.name));

    btn.disabled = true; setLoader(true);
    document.getElementById('cmpWrap').classList.add('hidden');
    try {
      const r = await fetch('/api/compare', { method: 'POST', body: fd });
      if (r.status === 401) { window.location.href = '/login'; return; }
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || 'Compare failed');
      document.getElementById('baselineList').innerHTML = j.baseline.map(renderMatch).join('');
      document.getElementById('newList').innerHTML     = j.new_model.map(renderMatch).join('');
      document.getElementById('cmpWrap').classList.remove('hidden');
    } catch (e) {
      alertBox(e.message);
    } finally {
      btn.disabled = false; setLoader(false);
    }
  });
}

window.initAdmin = initAdmin;
window.initCompare = initCompare;
