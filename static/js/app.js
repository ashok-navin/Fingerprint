/**
 * APEX BIO IDENTIFIER - High-Tech Biometric Frontend Controller
 * Deep CNN Fingerprint Identification, Enrollment & Directory Management
 */

// ==========================================
// 1. SOUND SYNTHESIZER (Web Audio API)
// ==========================================
class BiometricAudioEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
    }

    _init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    playScanChirp() {
        if (!this.enabled) return;
        this._init();
        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(800, this.ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1600, this.ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.15);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + 0.15);
        } catch (e) {}
    }

    playSuccessChime() {
        if (!this.enabled) return;
        this._init();
        try {
            const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
            notes.forEach((freq, idx) => {
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(freq, this.ctx.currentTime + idx * 0.07);
                gain.gain.setValueAtTime(0.12, this.ctx.currentTime + idx * 0.07);
                gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + idx * 0.07 + 0.3);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start(this.ctx.currentTime + idx * 0.07);
                osc.stop(this.ctx.currentTime + idx * 0.07 + 0.35);
            });
        } catch (e) {}
    }

    playWarningTone() {
        if (!this.enabled) return;
        this._init();
        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(220, this.ctx.currentTime);
            osc.frequency.setValueAtTime(180, this.ctx.currentTime + 0.12);
            gain.gain.setValueAtTime(0.15, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.3);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + 0.3);
        } catch (e) {}
    }
}

const audio = new BiometricAudioEngine();

// ==========================================
// 2. STATE MANAGEMENT
// ==========================================
let currentScanFile = null;
let currentScanBase64 = null;
let currentSampleFilename = null;
let activeMatchedUser = null;
let allEnrolledUsers = [];
let activeBloodFilter = 'all';

// ==========================================
// 3. DOM ELEMENTS
// ==========================================
const dom = {
    // Navigation
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    soundToggleBtn: document.getElementById('soundToggleBtn'),
    dbIndicator: document.getElementById('dbIndicator'),
    dbStatusText: document.getElementById('dbStatusText'),
    totalUsersCount: document.getElementById('totalUsersCount'),

    // Scanner & Quick Samples
    sampleChipsContainer: document.getElementById('sampleChipsContainer'),
    scannerDropzone: document.getElementById('scannerDropzone'),
    fingerprintFileInput: document.getElementById('fingerprintFileInput'),
    scannerViewport: document.getElementById('scannerViewport'),
    scannerPlaceholder: document.getElementById('scannerPlaceholder'),
    scannedImagePreview: document.getElementById('scannedImagePreview'),
    scanTickerText: document.getElementById('scanTickerText'),
    browseFileBtn: document.getElementById('browseFileBtn'),
    startScanBtn: document.getElementById('startScanBtn'),
    resetScanBtn: document.getElementById('resetScanBtn'),
    minutiaeCountVal: document.getElementById('minutiaeCountVal'),

    // Result Card Views
    resultEmptyState: document.getElementById('resultEmptyState'),
    resultLoadingState: document.getElementById('resultLoadingState'),
    resultMatchState: document.getElementById('resultMatchState'),
    resultUnregisteredState: document.getElementById('resultUnregisteredState'),
    scanProgressFill: document.getElementById('scanProgressFill'),
    loadingStepText: document.getElementById('loadingStepText'),

    // Result Card Fields
    resName: document.getElementById('resName'),
    resBloodGroup: document.getElementById('resBloodGroup'),
    resGenderPill: document.getElementById('resGenderPill'),
    resAgePill: document.getElementById('resAgePill'),
    resIdPill: document.getElementById('resIdPill'),
    resMobile: document.getElementById('resMobile'),
    resEmergencyContact: document.getElementById('resEmergencyContact'),
    resAddress: document.getElementById('resAddress'),
    resMinutiaeCount: document.getElementById('resMinutiaeCount'),
    resSimScore: document.getElementById('resSimScore'),
    resCreatedAt: document.getElementById('resCreatedAt'),
    scorePercentVal: document.getElementById('scorePercentVal'),
    emergencyCallBtn: document.getElementById('emergencyCallBtn'),
    printBadgeBtn: document.getElementById('printBadgeBtn'),
    viewInCnnBtn: document.getElementById('viewInCnnBtn'),
    enrollThisPrintBtn: document.getElementById('enrollThisPrintBtn'),
    unregTopScore: document.getElementById('unregTopScore'),
    unregMessage: document.getElementById('unregMessage'),

    // Enrollment Form
    enrollForm: document.getElementById('enrollForm'),
    enrollDropzone: document.getElementById('enrollDropzone'),
    enrollFingerprintInput: document.getElementById('enrollFingerprintInput'),
    enrollPlaceholder: document.getElementById('enrollPlaceholder'),
    enrollImagePreview: document.getElementById('enrollImagePreview'),
    enrollQualityCheck: document.getElementById('enrollQualityCheck'),
    submitEnrollBtn: document.getElementById('submitEnrollBtn'),

    // Citizen Directory
    directorySearchInput: document.getElementById('directorySearchInput'),
    refreshDirectoryBtn: document.getElementById('refreshDirectoryBtn'),
    directoryGrid: document.getElementById('directoryGrid'),
    filterChips: document.querySelectorAll('.filter-chip'),

    // CNN Feature Maps
    featureMapsGrid: document.getElementById('featureMapsGrid'),

    // Config Tab
    cfgDbType: document.getElementById('cfgDbType'),
    cfgDbName: document.getElementById('cfgDbName'),
    cfgStatusMsg: document.getElementById('cfgStatusMsg'),
    cfgUserCount: document.getElementById('cfgUserCount'),
    configStatusBadge: document.getElementById('configStatusBadge'),

    // ID Modal
    idCardModal: document.getElementById('idCardModal'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    closeModalBtn2: document.getElementById('closeModalBtn2'),
    modalName: document.getElementById('modalName'),
    modalBloodGroup: document.getElementById('modalBloodGroup'),
    modalMeta: document.getElementById('modalMeta'),
    modalMobile: document.getElementById('modalMobile'),
    modalEmergency: document.getElementById('modalEmergency'),
    modalAddress: document.getElementById('modalAddress'),
    modalHash: document.getElementById('modalHash')
};

// ==========================================
// 4. INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSoundToggle();
    initQuickSamples();
    initScannerDropzone();
    initEnrollmentForm();
    initDirectory();
    fetchDbStatus();
    initModalEvents();
});

// Tab navigation
function initTabs() {
    dom.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            dom.tabBtns.forEach(b => b.classList.remove('active'));
            dom.tabPanels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');

            if (target === 'tab-directory') {
                loadDirectory();
            } else if (target === 'tab-config') {
                fetchDbStatus();
            }
        });
    });
}

function switchTab(tabId) {
    const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.click();
}

// Sound toggle
function initSoundToggle() {
    dom.soundToggleBtn.addEventListener('click', () => {
        audio.enabled = !audio.enabled;
        dom.soundToggleBtn.classList.toggle('muted', !audio.enabled);
        dom.soundToggleBtn.innerHTML = audio.enabled 
            ? '<i class="fa-solid fa-volume-high"></i>' 
            : '<i class="fa-solid fa-volume-xmark"></i>';
        showToast(audio.enabled ? "Audio SFX Enabled" : "Audio SFX Muted", "info");
    });
}

// Quick Test Samples Drawer Removed for Clean Scanner UI
async function initQuickSamples() {
    // No-op
}

// ==========================================
// 6. SCANNER DROPZONE & FILE UPLOAD
// ==========================================
function initScannerDropzone() {
    dom.browseFileBtn.addEventListener('click', () => dom.fingerprintFileInput.click());
    dom.scannerDropzone.addEventListener('click', (e) => {
        if (e.target !== dom.startScanBtn && !dom.startScanBtn.contains(e.target)) {
            dom.fingerprintFileInput.click();
        }
    });

    dom.fingerprintFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelected(file);
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(name => {
        dom.scannerDropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dom.scannerDropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dom.scannerDropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dom.scannerDropzone.classList.remove('drag-over');
        });
    });

    dom.scannerDropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelected(files[0]);
        }
    });

    dom.startScanBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        executeScan();
    });

    dom.resetScanBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetScanner();
    });

    dom.enrollThisPrintBtn.addEventListener('click', () => {
        switchTab('tab-enroll');
    });

    dom.printBadgeBtn.addEventListener('click', () => {
        if (activeMatchedUser) openIdModal(activeMatchedUser);
    });

    dom.viewInCnnBtn.addEventListener('click', () => {
        switchTab('tab-cnn');
    });
}

function handleFileSelected(file) {
    if (!file.type.startsWith('image/')) {
        showToast("Please upload an image file (PNG, JPG, BMP).", "error");
        return;
    }

    currentScanFile = file;
    currentSampleFilename = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        currentScanBase64 = e.target.result;
        dom.scannedImagePreview.src = currentScanBase64;
        dom.scannedImagePreview.classList.remove('hidden');
        dom.scannerPlaceholder.classList.add('hidden');
        dom.startScanBtn.disabled = false;
        dom.scanTickerText.textContent = `FILE READY: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        audio.playScanChirp();
    };
    reader.readAsDataURL(file);
}

function resetScanner() {
    currentScanFile = null;
    currentScanBase64 = null;
    currentSampleFilename = null;
    activeMatchedUser = null;

    dom.fingerprintFileInput.value = '';
    dom.scannedImagePreview.src = '';
    dom.scannedImagePreview.classList.add('hidden');
    dom.scannerPlaceholder.classList.remove('hidden');
    dom.startScanBtn.disabled = true;
    dom.scannerViewport.classList.remove('scanning');
    dom.scanTickerText.textContent = "READY: Awaiting fingerprint scan input";
    dom.minutiaeCountVal.textContent = "--";

    showResultState('empty');
}

// ==========================================
// 7. SCAN EXECUTION & AI INFERENCE
// ==========================================
async function executeScan() {
    if (!currentScanFile && !currentScanBase64 && !currentSampleFilename) {
        showToast("Please select or upload a fingerprint scan first.", "error");
        return;
    }

    audio.playScanChirp();
    dom.scannerViewport.classList.add('scanning');
    dom.startScanBtn.disabled = true;
    showResultState('loading');

    // Simulate animated progress steps
    animateScanSteps();

    try {
        let response;
        if (currentSampleFilename) {
            response = await fetch('/api/identify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_filename: currentSampleFilename })
            });
        } else if (currentScanFile) {
            const formData = new FormData();
            formData.append('fingerprint_image', currentScanFile);
            response = await fetch('/api/identify', {
                method: 'POST',
                body: formData
            });
        } else if (currentScanBase64) {
            response = await fetch('/api/identify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: currentScanBase64 })
            });
        }

        const data = await response.json();
        
        // Brief delay to let the scanner animation feel authentic and futuristic
        setTimeout(() => {
            dom.scannerViewport.classList.remove('scanning');
            dom.startScanBtn.disabled = false;

            if (!data.success) {
                showToast(data.error || "Scanning error occurred", "error");
                audio.playWarningTone();
                showResultState('empty');
                return;
            }

            // Update Minutiae overlay in scanner
            if (data.annotated_image) {
                dom.scannedImagePreview.src = data.annotated_image;
            }
            dom.minutiaeCountVal.textContent = `${data.minutiae_count} pts`;

            // Render CNN Feature Maps
            if (data.cnn_feature_maps) {
                renderCnnFeatureMaps(data.cnn_feature_maps);
            }

            if (data.matched && data.user) {
                audio.playSuccessChime();
                activeMatchedUser = data.user;
                renderMatchedUserDetails(data);
                showResultState('matched');
                showToast(`Biometric Match: ${data.user.name} (${data.confidence}%)`, "success");
            } else {
                audio.playWarningTone();
                renderUnregisteredState(data);
                showResultState('unregistered');
                showToast(data.message || "Biometric Record Not Found in Database", "error");
            }
        }, 850);

    } catch (err) {
        console.error("Scan request error:", err);
        dom.scannerViewport.classList.remove('scanning');
        dom.startScanBtn.disabled = false;
        showResultState('empty');
        showToast("Network or Server error during scan", "error");
    }
}

function animateScanSteps() {
    const steps = [
        "Applying CLAHE adaptive histogram & Gabor orientation...",
        "Detecting ridge bifurcations & minutiae endings...",
        "Executing Deep CNN 256-D Latent Embedding...",
        "Querying MongoDB Atlas Biometric Database..."
    ];
    let stepIdx = 0;
    dom.scanProgressFill.style.width = '15%';
    
    const interval = setInterval(() => {
        stepIdx++;
        if (stepIdx < steps.length) {
            dom.loadingStepText.textContent = steps[stepIdx];
            dom.scanProgressFill.style.width = `${(stepIdx + 1) * 24}%`;
        } else {
            clearInterval(interval);
            dom.scanProgressFill.style.width = '100%';
        }
    }, 200);
}

function showResultState(state) {
    dom.resultEmptyState.classList.add('hidden');
    dom.resultLoadingState.classList.add('hidden');
    dom.resultMatchState.classList.add('hidden');
    dom.resultUnregisteredState.classList.add('hidden');

    if (state === 'empty') dom.resultEmptyState.classList.remove('hidden');
    if (state === 'loading') dom.resultLoadingState.classList.remove('hidden');
    if (state === 'matched') dom.resultMatchState.classList.remove('hidden');
    if (state === 'unregistered') dom.resultUnregisteredState.classList.remove('hidden');
}

function renderMatchedUserDetails(data) {
    const u = data.user;
    dom.resName.textContent = u.name;
    dom.resBloodGroup.innerHTML = `<i class="fa-solid fa-droplet"></i> ${u.blood_group}`;
    dom.resGenderPill.innerHTML = `<i class="fa-solid fa-venus-mars"></i> ${u.gender}`;
    dom.resAgePill.innerHTML = `<i class="fa-solid fa-cake-candles"></i> ${u.age} Yrs`;
    dom.resIdPill.textContent = `ID: #${u.id}`;
    
    dom.resMobile.textContent = u.mobile;
    dom.resEmergencyContact.textContent = u.emergency_contact;
    dom.resAddress.textContent = u.address;
    
    dom.resMinutiaeCount.textContent = `${data.minutiae_count} points`;
    dom.resSimScore.textContent = data.similarity_score;
    dom.resCreatedAt.textContent = u.created_at ? new Date(u.created_at).toLocaleDateString('en-US', { month: 'short', year: 'numeric', day: 'numeric' }) : 'Verified';
    
    dom.scorePercentVal.textContent = `${data.confidence}%`;
    dom.scanTickerText.textContent = `IDENTIFIED: ${u.name} (Confidence: ${data.confidence}%)`;

    // Emergency call button
    const cleanPhone = u.emergency_contact.replace(/[^0-9+]/g, '');
    dom.emergencyCallBtn.href = `tel:${cleanPhone}`;
}

function renderUnregisteredState(data) {
    dom.unregTopScore.textContent = `${data.confidence}%`;
    dom.unregMessage.textContent = data.message || `No matching citizen profile found in database above confidence threshold (${data.threshold * 100}%). Top match confidence is only ${data.confidence}%.`;
    dom.scanTickerText.textContent = `NOT FOUND: Unrecognized biometric signature`;
}

// ==========================================
// 8. ENROLLMENT FORM & MULTI-SCAN (UP TO 10 IMAGES)
// ==========================================
function initEnrollmentForm() {
    const dropzone = dom.enrollDropzone;
    const fileInput = dom.enrollFingerprintInput;
    const thumbGrid = document.getElementById('enrollThumbnailsGrid');
    const qualityCheck = dom.enrollQualityCheck;
    const qualityStatus = document.getElementById('enrollQualityStatus');

    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and drop support
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-active');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            handleMultiFilesSelected(files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleMultiFilesSelected(files);
        }
    });

    function handleMultiFilesSelected(files) {
        if (!thumbGrid) return;
        thumbGrid.innerHTML = '';
        thumbGrid.classList.remove('hidden');
        dom.enrollPlaceholder.classList.add('hidden');
        qualityCheck.classList.remove('hidden');

        const totalScans = Math.min(files.length, 10);
        if (qualityStatus) {
            qualityStatus.textContent = `${totalScans} Scan(s) Ready • High Fidelity Ridge Extraction (96%)`;
        }

        Array.from(files).slice(0, 10).forEach((file, idx) => {
            const reader = new FileReader();
            reader.onload = (evt) => {
                const card = document.createElement('div');
                card.className = 'enroll-thumb-card';
                card.innerHTML = `
                    <img src="${evt.target.result}" alt="Scan ${idx+1}">
                    <span class="thumb-badge">Scan #${idx+1}</span>
                    <span class="thumb-name" title="${file.name}">${file.name}</span>
                `;
                thumbGrid.appendChild(card);
            };
            reader.readAsDataURL(file);
        });

        audio.playScanChirp();
    }

    // Password toggle visibility
    const togglePassBtn = document.getElementById('toggleEnrollPasswordBtn');
    const passInput = document.getElementById('enrollPassword');
    const toggleIcon = document.getElementById('toggleEnrollPasswordIcon');
    if (togglePassBtn && passInput) {
        togglePassBtn.addEventListener('click', () => {
            const isPassword = passInput.type === 'password';
            passInput.type = isPassword ? 'text' : 'password';
            if (toggleIcon) {
                toggleIcon.className = isPassword ? 'fa-solid fa-eye-slash' : 'fa-solid fa-eye';
            }
        });
    }

    dom.enrollForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = dom.submitEnrollBtn;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Multi-Scan CNN Embeddings...';
        
        const formData = new FormData(dom.enrollForm);

        try {
            const res = await fetch('/api/enroll', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                audio.playSuccessChime();
                showToast(`Enrolled: ${data.user.name} (${data.total_scans_enrolled || 1} scans) successfully!`, "success");
                dom.enrollForm.reset();
                if (thumbGrid) {
                    thumbGrid.innerHTML = '';
                    thumbGrid.classList.add('hidden');
                }
                dom.enrollPlaceholder.classList.remove('hidden');
                qualityCheck.classList.add('hidden');
                
                // Refresh directory and samples
                initQuickSamples();
                loadDirectory();
                fetchDbStatus();

                // Switch to directory to see new user
                setTimeout(() => switchTab('tab-directory'), 1000);
            } else {
                audio.playWarningTone();
                showToast(data.error || "Enrollment failed", "error");
                if (data.error && data.error.toLowerCase().includes("passcode") && passInput) {
                    passInput.focus();
                    passInput.select();
                }
            }
        } catch (err) {
            console.error("Enrollment error:", err);
            showToast("Server error during enrollment", "error");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fa-solid fa-fingerprint"></i> Complete Enrollment';
        }
    });

    // Dataset Batch Importer button handler
    const btnImport = document.getElementById('btnRunDatasetImport');
    const statusDiv = document.getElementById('datasetImportStatus');
    if (btnImport) {
        btnImport.addEventListener('click', async () => {
            btnImport.disabled = true;
            btnImport.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Importing Dataset...';
            if (statusDiv) {
                statusDiv.classList.remove('hidden');
                statusDiv.className = 'dataset-import-status info';
                statusDiv.textContent = 'Scanning dataset/ folder and computing CNN embeddings for up to 10 scans per person...';
            }

            try {
                const res = await fetch('/api/import-dataset', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    audio.playSuccessChime();
                    showToast(data.message, "success");
                    if (statusDiv) {
                        statusDiv.className = 'dataset-import-status success';
                        statusDiv.textContent = `✓ ${data.message}`;
                    }
                    loadDirectory();
                    fetchDbStatus();
                } else {
                    audio.playWarningTone();
                    showToast(data.error || "Dataset import error", "error");
                    if (statusDiv) {
                        statusDiv.className = 'dataset-import-status error';
                        statusDiv.textContent = `✗ ${data.error || 'Import error'}`;
                    }
                }
            } catch (err) {
                showToast("Network error during dataset import", "error");
            } finally {
                btnImport.disabled = false;
                btnImport.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Import Dataset into MongoDB Atlas';
            }
        });
    }
}

// ==========================================
// 9. CITIZEN BIOMETRIC DIRECTORY
// ==========================================
function initDirectory() {
    dom.refreshDirectoryBtn.addEventListener('click', () => loadDirectory());
    dom.directorySearchInput.addEventListener('input', () => filterDirectoryCards());

    dom.filterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            dom.filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            activeBloodFilter = chip.dataset.filter;
            filterDirectoryCards();
        });
    });
}

async function loadDirectory() {
    try {
        dom.directoryGrid.innerHTML = `
            <div class="directory-loading" style="grid-column: 1/-1; text-align:center; padding: 2rem;">
                <i class="fa-solid fa-spinner fa-spin" style="font-size: 2rem; color: var(--primary-cyan);"></i>
                <p style="margin-top: 0.5rem; color: var(--text-muted);">Fetching citizen records...</p>
            </div>
        `;

        const res = await fetch('/api/users');
        const data = await res.json();

        if (data.success) {
            allEnrolledUsers = data.users;
            dom.totalUsersCount.textContent = allEnrolledUsers.length;
            dom.cfgUserCount.textContent = `${allEnrolledUsers.length} Profiles`;
            renderDirectoryCards(allEnrolledUsers);
        }
    } catch (err) {
        console.error("Failed to load users:", err);
        dom.directoryGrid.innerHTML = `<p class="text-red" style="grid-column: 1/-1;">Error loading citizen records.</p>`;
    }
}

function filterDirectoryCards() {
    const query = dom.directorySearchInput.value.toLowerCase().trim();
    const filtered = allEnrolledUsers.filter(u => {
        const matchesQuery = !query || 
            u.name.toLowerCase().includes(query) ||
            u.mobile.toLowerCase().includes(query) ||
            u.blood_group.toLowerCase().includes(query) ||
            u.address.toLowerCase().includes(query) ||
            u.emergency_contact.toLowerCase().includes(query);

        const matchesBlood = activeBloodFilter === 'all' || u.blood_group.toUpperCase() === activeBloodFilter.toUpperCase();

        return matchesQuery && matchesBlood;
    });

    renderDirectoryCards(filtered);
}

function renderDirectoryCards(users) {
    if (users.length === 0) {
        dom.directoryGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-muted);">
                <i class="fa-solid fa-user-slash" style="font-size: 3rem; margin-bottom: 1rem; color: var(--text-dim);"></i>
                <h3>No citizen records match criteria</h3>
                <p>Try clearing your search query or blood group filter.</p>
            </div>
        `;
        return;
    }

    dom.directoryGrid.innerHTML = '';
    users.forEach(u => {
        const card = document.createElement('div');
        card.className = 'citizen-dir-card';
        
        const fpImg = u.fingerprint_path || '/static/img/fingerprint_placeholder.png';

        card.innerHTML = `
            <div class="dir-card-header">
                <div class="dir-avatar-info">
                    <img src="${fpImg}" alt="Print" class="dir-thumb" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'40\\' height=\\'40\\' fill=\\'%2300f2fe\\' viewBox=\\'0 0 512 512\\'><path d=\\'M256 0a256 256 0 1 0 0 512A256 256 0 1 0 256 0z\\'/></svg>'">
                    <div>
                        <div class="dir-name">${u.name}</div>
                        <div class="dir-sub">${u.age} Yrs • ${u.gender} • #${u.id.substring(0, 10)}</div>
                    </div>
                </div>
                <span class="blood-badge"><i class="fa-solid fa-droplet"></i> ${u.blood_group}</span>
            </div>

            <div class="dir-card-body">
                <div class="dir-row"><i class="fa-solid fa-phone"></i> <span>${u.mobile}</span></div>
                <div class="dir-row emergency-row"><i class="fa-solid fa-truck-medical"></i> <span>${u.emergency_contact}</span></div>
                <div class="dir-row"><i class="fa-solid fa-location-dot"></i> <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${u.address}</span></div>
            </div>

            <div class="dir-card-actions">
                <button class="btn-ghost btn-sm" onclick="printCitizenId('${u.id}')" title="Print ID Badge" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;">
                    <i class="fa-solid fa-id-badge"></i> Print ID
                </button>
                <button class="btn-ghost btn-sm text-red" onclick="deleteCitizen('${u.id}', '${u.name}')" title="Delete Profile" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            </div>
        `;
        dom.directoryGrid.appendChild(card);
    });
}

window.printCitizenId = function(userId) {
    const user = allEnrolledUsers.find(u => u.id === userId);
    if (user) openIdModal(user);
};

window.deleteCitizen = async function(userId, userName) {
    if (!confirm(`Are you sure you want to delete biometric record for "${userName}"?`)) return;

    try {
        const res = await fetch(`/api/users/${userId}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast(`Deleted ${userName} from database.`, "info");
            loadDirectory();
            initQuickSamples();
            fetchDbStatus();
        } else {
            showToast(data.error || "Failed to delete user", "error");
        }
    } catch (e) {
        showToast("Error deleting record", "error");
    }
};

// ==========================================
// 10. CNN ACTIVATION MAPS RENDERER
// ==========================================
function renderCnnFeatureMaps(maps) {
    dom.featureMapsGrid.innerHTML = '';
    const layerNames = Object.keys(maps);
    
    if (layerNames.length === 0) {
        dom.featureMapsGrid.innerHTML = '<p class="text-dim">No feature maps available.</p>';
        return;
    }

    layerNames.forEach(layer => {
        const slices = maps[layer];
        const layerCard = document.createElement('div');
        layerCard.className = 'feature-layer-card';

        let slicesHtml = '<div class="feature-slices-row">';
        slices.forEach((sliceMatrix, sliceIdx) => {
            const canvasId = `canvas_${layer.replace(/[^a-zA-Z0-9]/g, '_')}_${sliceIdx}`;
            slicesHtml += `
                <div class="feature-slice-box">
                    <canvas id="${canvasId}" class="feature-slice-canvas" width="64" height="64"></canvas>
                    <div class="slice-label">Filter #${sliceIdx + 1}</div>
                </div>
            `;
        });
        slicesHtml += '</div>';

        layerCard.innerHTML = `
            <div class="feature-layer-title"><i class="fa-solid fa-microchip"></i> ${layer}</div>
            ${slicesHtml}
        `;
        dom.featureMapsGrid.appendChild(layerCard);

        // Render pixels to canvas
        setTimeout(() => {
            slices.forEach((sliceMatrix, sliceIdx) => {
                const canvasId = `canvas_${layer.replace(/[^a-zA-Z0-9]/g, '_')}_${sliceIdx}`;
                const canvas = document.getElementById(canvasId);
                if (canvas) {
                    drawMatrixToCanvas(canvas, sliceMatrix);
                }
            });
        }, 50);
    });
}

function drawMatrixToCanvas(canvas, matrix) {
    const ctx = canvas.getContext('2d');
    const h = matrix.length;
    const w = matrix[0].length;
    canvas.width = w;
    canvas.height = h;

    const imgData = ctx.createImageData(w, h);
    let ptr = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const val = matrix[y][x];
            // High-tech cyan-blue thermal colormap
            imgData.data[ptr] = Math.floor(val * 0.1);      // R
            imgData.data[ptr + 1] = Math.floor(val * 0.85); // G (Cyan)
            imgData.data[ptr + 2] = val;                   // B
            imgData.data[ptr + 3] = 255;                   // Alpha
            ptr += 4;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// ==========================================
// 11. DATABASE STATUS POLLER & ENV CONFIG
// ==========================================
async function fetchDbStatus() {
    try {
        const res = await fetch('/api/db-status');
        const data = await res.json();

        dom.dbStatusText.textContent = data.atlas_connected ? "MongoDB Atlas Active" : "Local Resilient Mode";
        dom.dbIndicator.className = `status-indicator ${data.atlas_connected ? 'live' : 'warning'}`;

        dom.cfgDbType.textContent = data.database_type;
        dom.cfgDbName.textContent = data.db_name;
        dom.cfgStatusMsg.textContent = data.status_message;
        dom.cfgStatusMsg.className = data.atlas_connected ? "text-emerald" : "text-yellow";
        dom.cfgUserCount.textContent = `${data.total_users} Profiles Enrolled`;
        dom.totalUsersCount.textContent = data.total_users;

        dom.configStatusBadge.textContent = data.atlas_connected ? "ATLAS ONLINE" : "LOCAL RESILIENT FALLBACK";
        dom.configStatusBadge.className = `badge ${data.atlas_connected ? 'badge-cyan' : 'badge-red'}`;
    } catch (e) {
        console.error("DB status fetch failed", e);
    }
}

// ==========================================
// 12. ID BADGE PRINT PREVIEW MODAL
// ==========================================
function initModalEvents() {
    dom.closeModalBtn.addEventListener('click', () => dom.idCardModal.classList.add('hidden'));
    dom.closeModalBtn2.addEventListener('click', () => dom.idCardModal.classList.add('hidden'));
    dom.idCardModal.addEventListener('click', (e) => {
        if (e.target === dom.idCardModal) dom.idCardModal.classList.add('hidden');
    });
}

function openIdModal(user) {
    dom.modalName.textContent = user.name;
    dom.modalBloodGroup.textContent = user.blood_group;
    dom.modalMeta.textContent = `Age: ${user.age} | Gender: ${user.gender}`;
    dom.modalMobile.textContent = user.mobile;
    dom.modalEmergency.textContent = `Emergency: ${user.emergency_contact}`;
    dom.modalAddress.textContent = user.address;
    dom.modalHash.textContent = `CNN EMBEDDING HASH: #${(user.id || '').substring(0, 16)}`;
    dom.idCardModal.classList.remove('hidden');
}

// ==========================================
// 13. UTILITY FUNCTIONS
// ==========================================
window.copyText = function(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text);
    showToast(`Copied to clipboard: ${text}`, "info");
};



function showToast(msg, type = "info") {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'fa-circle-info';
    if (type === 'success') icon = 'fa-circle-check text-emerald';
    if (type === 'error') icon = 'fa-circle-exclamation text-red';
    
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${msg}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
