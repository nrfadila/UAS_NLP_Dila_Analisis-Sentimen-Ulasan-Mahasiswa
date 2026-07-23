document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initTheme();
});

function initClock() {
    const clockEl = document.getElementById('digital-clock');
    const dateEl = document.getElementById('digital-date');
    
    function update() {
        const now = new Date();
        if (clockEl) {
            clockEl.textContent = now.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        if (dateEl) {
            dateEl.textContent = now.toLocaleDateString('id-ID', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
        }
    }
    update();
    setInterval(update, 1000);
}

function initTheme() {
    const toggleBtn = document.getElementById('theme-toggle-btn');
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', target);
            localStorage.setItem('theme', target);
            showToast(`Mode ${target === 'dark' ? 'Gelap' : 'Terang'} diaktifkan`);
        });
    }
}

function showToast(message, type = 'info') {
    let toast = document.getElementById('app-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'app-toast';
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3500);
}

function analyzeSentiment() {
    const input = document.getElementById('predict-input');
    const resultPanel = document.getElementById('predict-result');
    
    if (!input || !input.value.trim()) {
        showToast('Mohon masukkan teks tanggapan mahasiswa terlebih dahulu!', 'warning');
        if (input) input.focus();
        return;
    }
    
    const text = input.value.trim();
    
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            renderPredictionResult(data.data);
            showToast('Analisis sentimen berhasil!');
        } else {
            showToast(data.message || 'Terjadi kesalahan!', 'error');
        }
    })
    .catch(err => {
        showToast('Gagal menghubungi server.', 'error');
    });
}

function renderPredictionResult(res) {
    const panel = document.getElementById('predict-result');
    if (!panel) return;
    
    panel.style.display = 'block';
    
    const labelEl = document.getElementById('res-label');
    const confEl = document.getElementById('res-confidence');
    const posBar = document.getElementById('bar-pos');
    const negBar = document.getElementById('bar-neg');
    const netBar = document.getElementById('bar-net');
    
    const posVal = document.getElementById('val-pos');
    const negVal = document.getElementById('val-neg');
    const netVal = document.getElementById('val-net');
    
    if (labelEl) {
        labelEl.textContent = res.sentiment.toUpperCase();
        labelEl.className = `badge badge-${res.sentiment}`;
    }
    
    if (confEl) confEl.textContent = `${res.confidence}%`;
    
    if (posBar) posBar.style.width = `${res.probabilities.positif}%`;
    if (negBar) negBar.style.width = `${res.probabilities.negatif}%`;
    if (netBar) netBar.style.width = `${res.probabilities.netral}%`;
    
    if (posVal) posVal.textContent = `${res.probabilities.positif}%`;
    if (negVal) negVal.textContent = `${res.probabilities.negatif}%`;
    if (netVal) netVal.textContent = `${res.probabilities.netral}%`;
}

function triggerTrainModel() {
    const btn = document.getElementById('btn-train');
    const progress = document.getElementById('train-progress');
    const progressBar = document.getElementById('progress-fill');
    
    if (btn) btn.disabled = true;
    if (progress) progress.style.display = 'block';
    if (progressBar) progressBar.style.width = '30%';
    
    fetch('/train', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (progressBar) progressBar.style.width = '100%';
        setTimeout(() => {
            if (btn) btn.disabled = false;
            if (progress) progress.style.display = 'none';
            if (data.status === 'success') {
                showToast(data.message);
                location.reload();
            } else {
                showToast(data.message, 'error');
            }
        }, 500);
    })
    .catch(err => {
        if (btn) btn.disabled = false;
        if (progress) progress.style.display = 'none';
        showToast('Gagal melatih model.', 'error');
    });
}

function triggerResetModel() {
    if (!confirm('Apakah Anda yakin ingin mereset model ke kondisi awal?')) return;
    
    fetch('/reset', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(data.message);
            location.reload();
        } else {
            showToast(data.message, 'error');
        }
    });
}

function handleCsvUpload(input) {
    if (!input.files || !input.files[0]) return;
    
    const formData = new FormData();
    formData.append('file', input.files[0]);
    
    showToast('Mengunggah dan memproses dataset...');
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(data.message);
            setTimeout(() => location.reload(), 1200);
        } else {
            showToast(data.message, 'error');
        }
    })
    .catch(err => {
        showToast('Gagal mengunggah file.', 'error');
    });
}

function filterTable() {
    const searchInput = document.getElementById('search-input');
    const filterSelect = document.getElementById('sentiment-filter');
    const table = document.getElementById('dataset-table');
    
    if (!table) return;
    
    const query = searchInput ? searchInput.value.toLowerCase() : '';
    const filter = filterSelect ? filterSelect.value.toLowerCase() : 'all';
    
    const rows = table.querySelectorAll('tbody tr');
    let visibleCount = 0;
    
    rows.forEach(row => {
        const textCell = row.cells[1] ? row.cells[1].textContent.toLowerCase() : '';
        const sentimentCell = row.cells[2] ? row.cells[2].textContent.trim().toLowerCase() : '';
        
        const matchesSearch = textCell.includes(query);
        const matchesFilter = filter === 'all' || sentimentCell === filter;
        
        if (matchesSearch && matchesFilter) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });
    
    const countEl = document.getElementById('table-count-label');
    if (countEl) countEl.textContent = `Menampilkan ${visibleCount} data`;
}
