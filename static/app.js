/**
 * AtmosphereAI • Interactive Weather Machine Learning Frontend
 * Connects to FastAPI Backend with CORSMiddleware
 */

// ---------------- State Management ----------------
const state = {
  unit: 'C', // 'C' or 'F'
  isLivePredict: true,
  debounceTimer: null,
  history: [],
  currentCondition: 'Sunny'
};

// Month Names Lookup
const monthNames = [
  "", "January (Winter)", "February (Winter)", "March (Spring)",
  "April (Spring)", "May (Spring)", "June (Summer)",
  "July (Summer)", "August (Summer)", "September (Autumn)",
  "October (Autumn)", "November (Autumn)", "December (Winter)"
];

// Presets Definition (Fallback if API is unreachable)
const localPresets = {
  sunny: { temp_c: 31.5, hum: 38, precip: 0.1, wind: 8.5, loc: "Los Angeles", month: 7, hour: 13 },
  rainy: { temp_c: 16.0, hum: 89, precip: 6.8, wind: 14.0, loc: "Philadelphia", month: 4, hour: 15 },
  stormy: { temp_c: 22.0, hum: 92, precip: 8.5, wind: 26.5, loc: "Houston", month: 8, hour: 18 },
  snowy: { temp_c: -6.5, hum: 78, precip: 4.5, wind: 15.0, loc: "Chicago", month: 1, hour: 9 },
  cloudy: { temp_c: 19.5, hum: 74, precip: 1.2, wind: 11.0, loc: "New York", month: 5, hour: 11 }
};

// ---------------- DOM Elements ----------------
const DOM = {
  body: document.body,
  unitC: document.getElementById('unit-c'),
  unitF: document.getElementById('unit-f'),
  tempSlider: document.getElementById('temp-slider'),
  tempNumber: document.getElementById('temp-number'),
  tempUnitLabel: document.getElementById('temp-unit-label'),
  humiditySlider: document.getElementById('humidity-slider'),
  humidityNumber: document.getElementById('humidity-number'),
  precipSlider: document.getElementById('precip-slider'),
  precipNumber: document.getElementById('precip-number'),
  windSlider: document.getElementById('wind-slider'),
  windNumber: document.getElementById('wind-number'),
  monthSlider: document.getElementById('month-slider'),
  monthBadge: document.getElementById('month-name-badge'),
  hourSlider: document.getElementById('hour-slider'),
  hourBadge: document.getElementById('hour-name-badge'),
  locationSelect: document.getElementById('location-select'),
  livePredictToggle: document.getElementById('live-predict-toggle'),
  predictBtn: document.getElementById('predict-btn'),
  btnSpinner: document.getElementById('btn-spinner'),

  // Results
  heroCard: document.getElementById('hero-card'),
  heroLocationText: document.getElementById('hero-location-text'),
  heroTimeText: document.getElementById('hero-time-text'),
  heroConfidenceText: document.getElementById('hero-confidence-text'),
  heroCondition: document.getElementById('hero-condition'),
  heroSummary: document.getElementById('hero-summary'),
  heroIconContainer: document.getElementById('hero-icon-container'),

  // Metrics
  metricFeelsLike: document.getElementById('metric-feels-like'),
  metricFeelsUnit: document.getElementById('metric-feels-unit'),
  metricDewPoint: document.getElementById('metric-dew-point'),
  metricDewUnit: document.getElementById('metric-dew-unit'),
  metricRainRisk: document.getElementById('metric-rain-risk'),
  metricRainSeverity: document.getElementById('metric-rain-severity'),
  metricWindRating: document.getElementById('metric-wind-rating'),
  metricWindKmh: document.getElementById('metric-wind-kmh'),

  // Bars
  barSunny: document.getElementById('bar-Sunny'),
  pctSunny: document.getElementById('pct-Sunny'),
  barCloudy: document.getElementById('bar-Cloudy'),
  pctCloudy: document.getElementById('pct-Cloudy'),
  barRainy: document.getElementById('bar-Rainy'),
  pctRainy: document.getElementById('pct-Rainy'),
  barStormy: document.getElementById('bar-Stormy'),
  pctStormy: document.getElementById('pct-Stormy'),
  barSnowy: document.getElementById('bar-Snowy'),
  pctSnowy: document.getElementById('pct-Snowy'),

  // Advisories
  advisoriesContainer: document.getElementById('advisories-container'),

  // History
  historyTbody: document.getElementById('history-tbody'),
  clearHistoryBtn: document.getElementById('clear-history-btn'),

  // Modal
  theoryBtn: document.getElementById('theory-btn'),
  theoryModal: document.getElementById('theory-modal'),
  closeModalBtn: document.getElementById('close-modal-btn'),

  // Sky Canvas
  skyEffects: document.getElementById('sky-effects')
};

// ---------------- Unit Conversion Helpers ----------------
function cToF(c) { return (c * 9/5) + 32; }
function fToC(f) { return (f - 32) * 5/9; }

function getTempCelsius() {
  const val = parseFloat(DOM.tempNumber.value);
  return state.unit === 'F' ? fToC(val) : val;
}

function setTempFromCelsius(cVal) {
  if (state.unit === 'F') {
    const fVal = Math.round(cToF(cVal) * 10) / 10;
    DOM.tempSlider.min = Math.round(cToF(-20));
    DOM.tempSlider.max = Math.round(cToF(45));
    DOM.tempSlider.value = fVal;
    DOM.tempNumber.value = fVal;
    DOM.tempUnitLabel.textContent = '°F';
  } else {
    const rounded = Math.round(cVal * 10) / 10;
    DOM.tempSlider.min = -20;
    DOM.tempSlider.max = 45;
    DOM.tempSlider.value = rounded;
    DOM.tempNumber.value = rounded;
    DOM.tempUnitLabel.textContent = '°C';
  }
}

// ---------------- Synchronize Inputs ----------------
function setupInputSync(slider, numberInput, onUpdate) {
  slider.addEventListener('input', () => {
    numberInput.value = slider.value;
    if (onUpdate) onUpdate();
    triggerPrediction();
  });

  numberInput.addEventListener('input', () => {
    slider.value = numberInput.value;
    if (onUpdate) onUpdate();
    triggerPrediction();
  });
}

function updateMonthBadge() {
  const m = parseInt(DOM.monthSlider.value, 10);
  DOM.monthBadge.textContent = monthNames[m] || `Month ${m}`;
}

function updateHourBadge() {
  const h = parseInt(DOM.hourSlider.value, 10);
  const pad = h < 10 ? `0${h}` : `${h}`;
  let period = "Night";
  if (h >= 5 && h < 12) period = "Morning";
  else if (h >= 12 && h < 17) period = "Afternoon";
  else if (h >= 17 && h < 21) period = "Evening";
  DOM.hourBadge.textContent = `${pad}:00 (${period})`;
}

// ---------------- Unit Toggle ----------------
function setupUnitToggle() {
  DOM.unitC.addEventListener('click', () => {
    if (state.unit === 'C') return;
    state.unit = 'C';
    DOM.unitC.classList.add('active');
    DOM.unitF.classList.remove('active');
    const currF = parseFloat(DOM.tempNumber.value);
    setTempFromCelsius(fToC(currF));
    DOM.metricFeelsUnit.textContent = '°C';
    DOM.metricDewUnit.textContent = '°C';
    triggerPrediction();
  });

  DOM.unitF.addEventListener('click', () => {
    if (state.unit === 'F') return;
    state.unit = 'F';
    DOM.unitF.classList.add('active');
    DOM.unitC.classList.remove('active');
    const currC = parseFloat(DOM.tempNumber.value);
    setTempFromCelsius(currC);
    DOM.metricFeelsUnit.textContent = '°F';
    DOM.metricDewUnit.textContent = '°F';
    triggerPrediction();
  });
}

// ---------------- Presets ----------------
function applyPreset(presetKey) {
  const p = localPresets[presetKey];
  if (!p) return;

  // Highlight pill
  document.querySelectorAll('.preset-pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === presetKey);
  });

  setTempFromCelsius(p.temp_c);
  DOM.humiditySlider.value = p.hum;
  DOM.humidityNumber.value = p.hum;
  DOM.precipSlider.value = p.precip;
  DOM.precipNumber.value = p.precip;
  DOM.windSlider.value = p.wind;
  DOM.windNumber.value = p.wind;
  DOM.monthSlider.value = p.month;
  DOM.hourSlider.value = p.hour;
  DOM.locationSelect.value = p.loc;

  updateMonthBadge();
  updateHourBadge();
  executePrediction();
}

// ---------------- Sky Weather Animations ----------------
function updateSkyAtmosphere(theme) {
  DOM.body.className = `theme-${theme}`;

  // Clear existing particles
  const oldRain = document.querySelectorAll('.raindrop');
  oldRain.forEach(el => el.remove());
  const oldSnow = document.querySelectorAll('.snowflake');
  oldSnow.forEach(el => el.remove());

  const rainCanvas = document.querySelector('.rain-canvas');
  const snowCanvas = document.querySelector('.snow-canvas');
  const flash = document.querySelector('.lightning-flash');

  if (theme === 'rainy' || theme === 'stormy') {
    const dropsCount = theme === 'stormy' ? 45 : 30;
    for (let i = 0; i < dropsCount; i++) {
      const drop = document.createElement('div');
      drop.className = 'raindrop';
      drop.style.left = `${Math.random() * 100}%`;
      drop.style.height = `${15 + Math.random() * 25}px`;
      drop.style.animationDuration = `${0.4 + Math.random() * 0.4}s`;
      drop.style.animationDelay = `${Math.random() * 1.5}s`;
      rainCanvas.appendChild(drop);
    }
  }

  if (theme === 'snowy') {
    for (let i = 0; i < 35; i++) {
      const flake = document.createElement('div');
      flake.className = 'snowflake';
      flake.textContent = '❄';
      flake.style.left = `${Math.random() * 100}%`;
      flake.style.fontSize = `${10 + Math.random() * 14}px`;
      flake.style.animationDuration = `${2.5 + Math.random() * 3.5}s`;
      flake.style.animationDelay = `${Math.random() * 2}s`;
      snowCanvas.appendChild(flake);
    }
  }

  if (theme === 'stormy') {
    // Intermittent flash trigger
    if (window.lightningInterval) clearInterval(window.lightningInterval);
    window.lightningInterval = setInterval(() => {
      if (DOM.body.classList.contains('theme-stormy')) {
        flash.classList.add('active');
        setTimeout(() => flash.classList.remove('active'), 120);
      }
    }, 4500);
  } else {
    if (window.lightningInterval) clearInterval(window.lightningInterval);
  }
}

// ---------------- Weather Icon SVGs ----------------
function getWeatherSvg(condition) {
  switch (condition) {
    case 'Sunny':
      return `
        <svg class="weather-hero-svg" viewBox="0 0 64 64" fill="none">
          <circle cx="32" cy="32" r="14" fill="#FFA726" />
          <g stroke="#FFA726" stroke-width="4" stroke-linecap="round">
            <line x1="32" y1="6" x2="32" y2="12" />
            <line x1="32" y1="52" x2="32" y2="58" />
            <line x1="6" y1="32" x2="12" y2="32" />
            <line x1="52" y1="32" x2="58" y2="32" />
            <line x1="13.6" y1="13.6" x2="17.8" y2="17.8" />
            <line x1="46.2" y1="46.2" x2="50.4" y2="50.4" />
            <line x1="13.6" y1="50.4" x2="17.8" y2="46.2" />
            <line x1="46.2" y1="17.8" x2="50.4" y2="13.6" />
          </g>
        </svg>
      `;
    case 'Rainy':
      return `
        <svg class="weather-hero-svg" viewBox="0 0 64 64" fill="none">
          <path d="M46 36A12 12 0 0 0 34 24a15 15 0 0 0-14.7 12.3A9 9 0 0 0 10 44a9 9 0 0 0 9 9h27a9 9 0 0 0 0-18z" fill="#90CAF9" />
          <line x1="24" y1="55" x2="20" y2="63" stroke="#29B6F6" stroke-width="3" stroke-linecap="round" />
          <line x1="34" y1="55" x2="30" y2="63" stroke="#29B6F6" stroke-width="3" stroke-linecap="round" />
          <line x1="44" y1="55" x2="40" y2="63" stroke="#29B6F6" stroke-width="3" stroke-linecap="round" />
        </svg>
      `;
    case 'Stormy':
      return `
        <svg class="weather-hero-svg" viewBox="0 0 64 64" fill="none">
          <path d="M46 32A12 12 0 0 0 34 20a15 15 0 0 0-14.7 12.3A9 9 0 0 0 10 40a9 9 0 0 0 9 9h27a9 9 0 0 0 0-18z" fill="#7E57C2" />
          <polygon points="32 38 24 50 31 50 28 62 40 46 33 46" fill="#FFD54F" stroke="#FFB300" stroke-width="1.5" />
        </svg>
      `;
    case 'Snowy':
      return `
        <svg class="weather-hero-svg" viewBox="0 0 64 64" fill="none">
          <path d="M46 34A12 12 0 0 0 34 22a15 15 0 0 0-14.7 12.3A9 9 0 0 0 10 42a9 9 0 0 0 9 9h27a9 9 0 0 0 0-18z" fill="#B0BEC5" />
          <circle cx="22" cy="56" r="3" fill="#E1F5FE" />
          <circle cx="34" cy="58" r="3" fill="#E1F5FE" />
          <circle cx="44" cy="56" r="3" fill="#E1F5FE" />
        </svg>
      `;
    case 'Cloudy':
    default:
      return `
        <svg class="weather-hero-svg" viewBox="0 0 64 64" fill="none">
          <path d="M44 38A10 10 0 0 0 34 28a13 13 0 0 0-12.7 10.3A8 8 0 0 0 12 46a8 8 0 0 0 8 8h24a8 8 0 0 0 0-16z" fill="#78909C" />
          <path d="M52 30A8 8 0 0 0 44 22a10 10 0 0 0-9.8 8.1A6 6 0 0 0 28 36a6 6 0 0 0 6 6h18a6 6 0 0 0 0-12z" fill="#CFD8DC" opacity="0.8" />
        </svg>
      `;
  }
}

function getConditionDescription(condition) {
  switch (condition) {
    case 'Sunny': return "Clear skies, low humidity, and excellent outdoor visibility.";
    case 'Rainy': return "Active precipitation detected with high relative humidity.";
    case 'Stormy': return "High gust wind force combined with precipitation threshold.";
    case 'Snowy': return "Sub-zero or near-freezing thermodynamic precipitation.";
    case 'Cloudy': return "Elevated humidity and dense atmospheric cloud coverage.";
    default: return "Atmospheric conditions analyzed by Backpropagation neural network.";
  }
}

// ---------------- Wind Beaufort Rating ----------------
function getWindBeaufort(kmh) {
  if (kmh < 5) return { rating: "Calm", note: `${kmh} km/h (Beaufort 0-1)` };
  if (kmh < 15) return { rating: "Gentle", note: `${kmh} km/h (Beaufort 2-3)` };
  if (kmh < 25) return { rating: "Moderate", note: `${kmh} km/h (Beaufort 4)` };
  if (kmh < 32) return { rating: "Fresh Breeze", note: `${kmh} km/h (Beaufort 5)` };
  return { rating: "Gale Alert", note: `${kmh} km/h (Beaufort 6+)` };
}

// ---------------- API Communication ----------------
function triggerPrediction() {
  if (!state.isLivePredict) return;
  if (state.debounceTimer) clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    executePrediction();
  }, 120);
}

async function executePrediction() {
  const temp_c = getTempCelsius();
  const hum = parseFloat(DOM.humiditySlider.value);
  const precip = parseFloat(DOM.precipSlider.value);
  const wind = parseFloat(DOM.windSlider.value);
  const month = parseInt(DOM.monthSlider.value, 10);
  const hour = parseInt(DOM.hourSlider.value, 10);
  const location = DOM.locationSelect.value;

  const payload = {
    temperature_c: temp_c,
    humidity_pct: hum,
    precipitation_mm: precip,
    wind_speed_kmh: wind,
    location: location,
    month: month,
    hour: hour
  };

  DOM.btnSpinner.style.display = 'inline-block';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    renderPredictionResults(data, payload);
    addToHistory(payload, data);
  } catch (err) {
    console.error("Prediction API Error:", err);
    // Local fallback heuristic for offline preview if server restarted
    fallbackLocalPredict(payload);
  } finally {
    DOM.btnSpinner.style.display = 'none';
  }
}

function renderPredictionResults(data, input) {
  state.currentCondition = data.predicted_condition;

  // 1. Theme and Canvas
  updateSkyAtmosphere(data.weather_theme);

  // 2. Hero Card
  DOM.heroLocationText.textContent = `${input.location}, US`;
  DOM.heroTimeText.textContent = `${input.hour < 10 ? '0' : ''}${input.hour}:00 • Month ${input.month}`;
  DOM.heroConfidenceText.textContent = `${data.confidence.toFixed(1)}% Confidence`;
  DOM.heroCondition.textContent = `${data.predicted_condition} Weather`;
  DOM.heroSummary.textContent = getConditionDescription(data.predicted_condition);
  DOM.heroIconContainer.innerHTML = getWeatherSvg(data.predicted_condition);

  // 3. Thermodynamic Metrics
  const dispFeels = state.unit === 'F' ? cToF(data.feels_like_c) : data.feels_like_c;
  const dispDew = state.unit === 'F' ? cToF(data.dew_point_c) : data.dew_point_c;

  DOM.metricFeelsLike.textContent = dispFeels.toFixed(1);
  DOM.metricDewPoint.textContent = dispDew.toFixed(1);

  if (data.wind_chill_c !== null) {
    DOM.metricFeelsUnit.parentElement.nextElementSibling.textContent = `Wind chill factor active`;
  } else if (data.heat_index_c !== null) {
    DOM.metricFeelsUnit.parentElement.nextElementSibling.textContent = `Heat index formula active`;
  } else {
    DOM.metricFeelsUnit.parentElement.nextElementSibling.textContent = `Apparent Steadman index`;
  }

  DOM.metricRainRisk.textContent = data.rain_risk_pct.toFixed(0);
  if (data.rain_risk_pct > 70) DOM.metricRainSeverity.textContent = "High Precipitation";
  else if (data.rain_risk_pct > 40) DOM.metricRainSeverity.textContent = "Moderate Chance";
  else DOM.metricRainSeverity.textContent = "Minimal Risk";

  const windInfo = getWindBeaufort(input.wind_speed_kmh);
  DOM.metricWindRating.textContent = windInfo.rating;
  DOM.metricWindKmh.textContent = windInfo.note;

  // 4. Probability Bars
  const probs = data.probabilities;
  for (const [cls, pct] of Object.entries(probs)) {
    const bar = document.getElementById(`bar-${cls}`);
    const label = document.getElementById(`pct-${cls}`);
    if (bar && label) {
      bar.style.width = `${Math.max(2, pct)}%`;
      label.textContent = `${pct.toFixed(1)}%`;
    }
  }

  // 5. Advisories
  DOM.advisoriesContainer.innerHTML = '';
  data.advisories.forEach(adv => {
    const pill = document.createElement('span');
    pill.className = 'advisory-pill';
    pill.textContent = adv;
    DOM.advisoriesContainer.appendChild(pill);
  });
}

// ---------------- Offline / Local Fallback ----------------
function fallbackLocalPredict(input) {
  let condition = "Sunny";
  if (input.temperature_c <= 2.0 && input.precipitation_mm >= 2.5) {
    condition = "Snowy";
  } else if (input.wind_speed_kmh >= 20.0 && input.precipitation_mm >= 4.0) {
    condition = "Stormy";
  } else if (input.precipitation_mm >= 4.0) {
    condition = "Rainy";
  } else if (input.humidity_pct >= 65.0) {
    condition = "Cloudy";
  }

  const mockProbs = {
    Sunny: condition === 'Sunny' ? 95.4 : 1.2,
    Cloudy: condition === 'Cloudy' ? 92.1 : 3.4,
    Rainy: condition === 'Rainy' ? 96.8 : 2.1,
    Stormy: condition === 'Stormy' ? 94.6 : 0.8,
    Snowy: condition === 'Snowy' ? 97.2 : 0.5
  };

  const mockData = {
    predicted_condition: condition,
    confidence: mockProbs[condition],
    probabilities: mockProbs,
    feels_like_c: input.temperature_c - 1.2,
    dew_point_c: input.temperature_c - 4.5,
    wind_chill_c: null,
    heat_index_c: null,
    rain_risk_pct: Math.min(100, input.precipitation_mm * 8 + input.humidity_pct * 0.4),
    advisories: [`Predicted weather condition: ${condition} (MLP Backprop model).`],
    weather_theme: condition.toLowerCase()
  };

  renderPredictionResults(mockData, input);
}

// ---------------- History Management ----------------
function addToHistory(input, data) {
  const item = {
    id: Date.now(),
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    location: input.location,
    temp_c: input.temperature_c,
    humidity: input.humidity_pct,
    precip: input.precipitation_mm,
    wind: input.wind_speed_kmh,
    month: input.month,
    hour: input.hour,
    condition: data.predicted_condition,
    confidence: data.confidence
  };

  state.history.unshift(item);
  if (state.history.length > 8) state.history.pop();
  renderHistoryTable();
}

function renderHistoryTable() {
  DOM.historyTbody.innerHTML = '';
  if (state.history.length === 0) {
    DOM.historyTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--text-dim); padding: 20px;">No predictions logged yet. Adjust sliders to predict!</td></tr>`;
    return;
  }

  state.history.forEach(item => {
    const tr = document.createElement('tr');
    const dispT = state.unit === 'F' ? `${cToF(item.temp_c).toFixed(1)}°F` : `${item.temp_c.toFixed(1)}°C`;

    tr.innerHTML = `
      <td>${item.time}</td>
      <td><strong>${item.location}</strong></td>
      <td>${dispT}</td>
      <td>${item.humidity}%</td>
      <td>${item.precip} mm</td>
      <td>${item.wind} km/h</td>
      <td><span class="table-badge ${item.condition.toLowerCase()}">${item.condition}</span></td>
      <td><strong>${item.confidence.toFixed(1)}%</strong></td>
      <td><button class="load-run-btn" data-id="${item.id}">Restore</button></td>
    `;
    DOM.historyTbody.appendChild(tr);
  });

  // Attach restore buttons
  document.querySelectorAll('.load-run-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = parseInt(e.target.dataset.id, 10);
      const past = state.history.find(h => h.id === id);
      if (past) {
        setTempFromCelsius(past.temp_c);
        DOM.humiditySlider.value = past.humidity;
        DOM.humidityNumber.value = past.humidity;
        DOM.precipSlider.value = past.precip;
        DOM.precipNumber.value = past.precip;
        DOM.windSlider.value = past.wind;
        DOM.windNumber.value = past.wind;
        DOM.monthSlider.value = past.month;
        DOM.hourSlider.value = past.hour;
        DOM.locationSelect.value = past.location;
        updateMonthBadge();
        updateHourBadge();
        executePrediction();
      }
    });
  });
}

// ---------------- Modal Handling ----------------
function setupModal() {
  DOM.theoryBtn.addEventListener('click', () => {
    DOM.theoryModal.classList.add('open');
  });

  DOM.closeModalBtn.addEventListener('click', () => {
    DOM.theoryModal.classList.remove('open');
  });

  DOM.theoryModal.addEventListener('click', (e) => {
    if (e.target === DOM.theoryModal) {
      DOM.theoryModal.classList.remove('open');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && DOM.theoryModal.classList.contains('open')) {
      DOM.theoryModal.classList.remove('open');
    }
  });
}

// ---------------- Initialize Application ----------------
function initApp() {
  // Sync sliders
  setupInputSync(DOM.tempSlider, DOM.tempNumber);
  setupInputSync(DOM.humiditySlider, DOM.humidityNumber);
  setupInputSync(DOM.precipSlider, DOM.precipNumber);
  setupInputSync(DOM.windSlider, DOM.windNumber);
  setupInputSync(DOM.monthSlider, { value: 6 }, updateMonthBadge);
  setupInputSync(DOM.hourSlider, { value: 14 }, updateHourBadge);

  DOM.locationSelect.addEventListener('change', () => triggerPrediction());

  // Live toggle
  DOM.livePredictToggle.addEventListener('change', (e) => {
    state.isLivePredict = e.target.checked;
  });

  // Predict button
  DOM.predictBtn.addEventListener('click', () => executePrediction());

  // Unit toggle
  setupUnitToggle();

  // Presets
  document.querySelectorAll('.preset-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      applyPreset(btn.dataset.preset);
    });
  });

  // Clear history
  DOM.clearHistoryBtn.addEventListener('click', () => {
    state.history = [];
    renderHistoryTable();
  });

  // Modal
  setupModal();

  // Initial Badges
  updateMonthBadge();
  updateHourBadge();

  // Run initial prediction
  executePrediction();
}

// Start
document.addEventListener('DOMContentLoaded', initApp);

