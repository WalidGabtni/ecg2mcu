#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "MAX30100_PulseOximeter.h"
#include "cnn1d_model.h"
#include "secrets.h"

// ── Credentials (WIFI_SSID, WIFI_PASSWORD, SUPABASE_URL, SUPABASE_KEY, DEVICE_ID) ──

// ── TFT ───────────────────────────────────────────────────────────────────────
#define TFT_CS   5
#define TFT_RST  4
#define TFT_DC   15
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// ── ADC / Lead-off ────────────────────────────────────────────────────────────
#define ECG_PIN 34
#define LO_PLUS  32
#define LO_MINUS 33
#define SDN_PIN  27

// ── MAX30100 ──────────────────────────────────────────────────────────────────
PulseOximeter pox;
bool maxReady       = false;
bool hasReading     = false;
bool poxUpdated     = false;
bool wifiActive     = false;
unsigned long lastPoxUpdate = 0;
unsigned long lastBeatMs    = 0;
#define MAX_SILENCE_MS 60000

// ── TFLite ────────────────────────────────────────────────────────────────────
#define TENSOR_ARENA_SIZE (80 * 1024)
uint8_t* tensor_arena = nullptr;
const tflite::Model* model_tfl = nullptr;
tflite::MicroInterpreter* interp = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ── Constants ─────────────────────────────────────────────────────────────────
const char* CLASS_NAMES[] = {"NORM", "MI", "STTC", "CD", "HYP"};
const float THRESHOLDS[]  = {0.496, 0.342, 0.254, 0.270, 0.197};

// ── Live acquisition window ───────────────────────────────────────────────────
#define WINDOW_LEN 1000        // 10 s @ 100 Hz, matches training input shape (1,1000,1)
#define SAMPLE_PERIOD_US 10000 // 100 Hz -> 10 ms per sample
float liveWindow[WINDOW_LEN];  // raw ADC values for the current 10 s window
float liveDisplay[160];        // downsampled copy for the TFT trace

// ── State ─────────────────────────────────────────────────────────────────────
String patientId              = "";
unsigned long lastVitalsPost  = 0;
unsigned long lastHistoryPost = 0;
int   currentHR             = 72;
int   currentSpo2           = 98;
int   currentRR             = 833;
char  currentPrediction[8]  = "NORM";
int   currentConfidence     = 0;

// ── Beat callback ─────────────────────────────────────────────────────────────
float lastValidHR    = 0;
int   validBeatCount = 0;
void onBeatDetected() {
  float hr   = pox.getHeartRate();
  float spo2 = pox.getSpO2();
  if (hr < 40 || hr > 180) { validBeatCount = 0; return; }
  if (lastValidHR > 0 && abs(hr - lastValidHR) > 40) {
    lastValidHR = hr; validBeatCount = 0; return;
  }
  lastValidHR = hr;
  validBeatCount++;
  if (validBeatCount < 3) return;
  Serial.printf("Beat! HR=%.1f SpO2=%.1f%%\n", hr, spo2);
  lastBeatMs = millis();
  currentHR  = (int)hr;
  currentRR  = 60000 / currentHR;
  hasReading = true;
  poxUpdated = true;
  if (spo2 > 80 && spo2 <= 100) currentSpo2 = (int)spo2;
}

// ── MAX30100 health check ─────────────────────────────────────────────────────
void checkMaxHealth() {
  if (!maxReady || wifiActive) return;
  if (millis() - lastBeatMs > MAX_SILENCE_MS) {
    Serial.println("MAX30100 silent — reinitializing...");
    if (pox.begin()) {
      pox.setOnBeatDetectedCallback(onBeatDetected);
      pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);
      Serial.println("MAX30100 recovered");
    } else {
      Serial.println("MAX30100 re-init failed");
    }
    lastBeatMs = millis();
  }
}

// ── WiFi ──────────────────────────────────────────────────────────────────────
void connectWiFi() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(2, 10);
  tft.setTextColor(ST77XX_YELLOW);
  tft.setTextSize(1);
  tft.println("Connecting WiFi...");
  tft.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500); tft.print("."); attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    tft.fillScreen(ST77XX_BLACK);
    tft.setCursor(2, 10);
    tft.setTextColor(ST77XX_GREEN);
    tft.println("WiFi Connected!");
    tft.println(WiFi.localIP().toString());
    Serial.println("WiFi connected: " + WiFi.localIP().toString());
    delay(1000);
  } else {
    tft.fillScreen(ST77XX_BLACK);
    tft.setCursor(2, 10);
    tft.setTextColor(ST77XX_RED);
    tft.println("WiFi Failed!");
    Serial.println("WiFi failed");
    delay(1000);
  }
}

// ── Fetch patient ID ──────────────────────────────────────────────────────────
void fetchPatientId() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(SUPABASE_URL) + "/rest/v1/patients?device_id=eq." + DEVICE_ID + "&select=id";
  http.begin(url);
  http.addHeader("apikey", SUPABASE_KEY);
  http.addHeader("Authorization", "Bearer " + String(SUPABASE_KEY));
  int code = http.GET();
  if (code == 200) {
    String payload = http.getString();
    JsonDocument doc;
    deserializeJson(doc, payload);
    if (doc.size() > 0) {
      patientId = doc[0]["id"].as<String>();
      Serial.println("Patient ID: " + patientId);
    }
  }
  http.end();
}

// ── Build ECG JSON from the live window ───────────────────────────────────────
String buildLiveEcgJson() {
  String json = "[";
  json.reserve(4200);
  for (int i = 0; i < 500; i++) {
    int idx = (i * WINDOW_LEN) / 500;
    json += String(liveWindow[idx], 3);
    if (i < 499) json += ",";
  }
  return json + "]";
}

// ── PATCH patients ────────────────────────────────────────────────────────────
void updatePatientVitals(const char* prediction, float confidence) {
  if (WiFi.status() != WL_CONNECTED) return;
  wifiActive = true;
  strncpy(currentPrediction, prediction, 7);
  currentConfidence = (int)(confidence * 100);
  HTTPClient http;
  String url = String(SUPABASE_URL) + "/rest/v1/patients?device_id=eq." + DEVICE_ID;
  http.begin(url);
  http.addHeader("apikey", SUPABASE_KEY);
  http.addHeader("Authorization", "Bearer " + String(SUPABASE_KEY));
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Prefer", "return=minimal");
  String body = "{\"connection_status\":\"connected\","
                "\"current_hr\":" + String(currentHR) + ","
                "\"current_spo2\":" + String(currentSpo2) + ","
                "\"current_rr\":" + String(currentRR) + ","
                "\"current_prediction\":\"" + String(prediction) + "\","
                "\"current_confidence\":" + String((int)(confidence * 100)) + ","
                "\"last_reading\":\"now()\","
                "\"ecg_buffer\":" + buildLiveEcgJson() + "}";
  int code = http.PATCH(body);
  Serial.printf("PATCH patients: %d\n", code);
  http.end();
  wifiActive = false;
}

// ── POST vitals_history ───────────────────────────────────────────────────────
void postVitalsHistory() {
  if (WiFi.status() != WL_CONNECTED || patientId == "") return;
  wifiActive = true;
  HTTPClient http;
  String url = String(SUPABASE_URL) + "/rest/v1/vitals_history";
  http.begin(url);
  http.addHeader("apikey", SUPABASE_KEY);
  http.addHeader("Authorization", "Bearer " + String(SUPABASE_KEY));
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Prefer", "return=minimal");
  JsonDocument doc;
  doc["patient_id"]  = patientId;
  doc["heart_rate"]  = currentHR;
  doc["spo2"]        = currentSpo2;
  doc["rr_interval"] = currentRR;
  String body;
  serializeJson(doc, body);
  int code = http.POST(body);
  Serial.printf("POST vitals_history: %d\n", code);
  http.end();
  wifiActive = false;
}

// ── Lead-off check ────────────────────────────────────────────────────────────
bool leadsAttached() {
  // AD8232 pulls LO+/LO- HIGH when an electrode is off
  return digitalRead(LO_PLUS) == LOW && digitalRead(LO_MINUS) == LOW;
}

// ── Acquire 10 s live window @ 100 Hz ─────────────────────────────────────────
// Samples ECG_PIN, updates the TFT trace live as it goes, and keeps MAX30100
// and WiFi housekeeping running during the 10 s acquisition.
// Returns false if a lead disconnect was detected at any point during the
// window — in that case the captured data is unreliable and must be discarded
// rather than fed to the model.
bool acquireLiveWindow() {
  tft.fillRect(0, 18, 160, 94, ST77XX_BLACK);
  for (int x = 0; x < 160; x += 16) tft.drawFastVLine(x, 18, 90, 0x1082);
  for (int y = 18; y < 108; y += 14) tft.drawFastHLine(0, y, 160, 0x1082);

  int prevY = 63;
  unsigned long nextSample = micros();
  bool leadOffDuringCapture = false;

  for (int i = 0; i < WINDOW_LEN; i++) {
    while ((long)(micros() - nextSample) < 0) {
      if (maxReady && !wifiActive && (millis() - lastPoxUpdate >= 20)) {
        lastPoxUpdate = millis();
        pox.update();
      }
    }
    nextSample += SAMPLE_PERIOD_US;

    if (!leadsAttached()) leadOffDuringCapture = true;

    liveWindow[i] = (float)analogRead(ECG_PIN);

    if (i % (WINDOW_LEN / 160) == 0) {
      int col = i / (WINDOW_LEN / 160);
      int curY = (int)(63 - ((liveWindow[i] - 2048.0f) / 2048.0f) * 40.0f);
      curY = constrain(curY, 20, 106);
      if (col > 0) tft.drawLine(col - 1, prevY, col, curY, leadOffDuringCapture ? ST77XX_RED : ST77XX_GREEN);
      prevY = curY;
    }
  }

  if (leadOffDuringCapture) {
    Serial.println("REJECTED window: lead-off detected during acquisition — discarding, not running inference");
    return false;
  }
  return true;
}

// ── Run inference on the live window (single channel, z-normalized) ───────────
void runLiveInference() {
  // z-normalize the window (mean 0, unit variance) — same as PTB-XL preprocessing
  float mean = 0;
  for (int i = 0; i < WINDOW_LEN; i++) mean += liveWindow[i];
  mean /= WINDOW_LEN;

  float var = 0;
  for (int i = 0; i < WINDOW_LEN; i++) {
    float d = liveWindow[i] - mean;
    var += d * d;
  }
  float stddev = sqrt(var / WINDOW_LEN);
  if (stddev < 1e-6f) stddev = 1e-6f; // guard against a flat/disconnected line

  // quantize into the model's INT8 input tensor (single channel: t*1)
  float input_scale      = input_tensor->params.scale;
  int   input_zero_point = input_tensor->params.zero_point;
  for (int t = 0; t < WINDOW_LEN; t++) {
    float znorm = (liveWindow[t] - mean) / stddev;
    int q = (int)roundf(znorm / input_scale + input_zero_point);
    q = constrain(q, -128, 127);
    input_tensor->data.int8[t] = (int8_t)q;
  }

  interp->Invoke();

  int best = 0; float best_prob = 0.0; bool any_det = false;
  Serial.println("\n[LIVE inference]");
  for (int i = 0; i < 5; i++) {
    float logit = (output_tensor->data.int8[i] - output_tensor->params.zero_point) * output_tensor->params.scale;
    float prob  = 1.0f / (1.0f + exp(-logit));
    bool  det   = prob >= THRESHOLDS[i];
    Serial.printf("  %s: %.4f %s\n", CLASS_NAMES[i], prob, det ? "DETECTED" : "");
    if (det && prob > best_prob) { best_prob = prob; best = i; any_det = true; }
  }
  if (!any_det)
    for (int i = 0; i < 5; i++) {
      float logit = (output_tensor->data.int8[i] - output_tensor->params.zero_point) * output_tensor->params.scale;
      float prob  = 1.0f / (1.0f + exp(-logit));
      if (prob > best_prob) { best_prob = prob; best = i; }
    }
  Serial.printf("Prediction: %s (%.2f%%)\n", CLASS_NAMES[best], best_prob * 100);

  tft.fillRect(0, 112, 160, 16, ST77XX_BLACK);
  tft.setCursor(2, 114);
  tft.setTextColor(leadsAttached() ? ST77XX_WHITE : ST77XX_RED);
  tft.setTextSize(1);
  tft.printf("%s | %.0f%%%s", CLASS_NAMES[best], best_prob * 100, leadsAttached() ? "" : " LO!");

  unsigned long now = millis();
  if (now - lastVitalsPost > 2000)   { updatePatientVitals(CLASS_NAMES[best], best_prob); lastVitalsPost = now; }
  if (now - lastHistoryPost > 30000) { postVitalsHistory(); lastHistoryPost = now; }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  pinMode(SDN_PIN, OUTPUT);
  digitalWrite(SDN_PIN, HIGH);

  Wire.begin(21, 22);
  delay(300);

  for (int attempt = 0; attempt < 5 && !maxReady; attempt++) {
    if (pox.begin()) {
      pox.setOnBeatDetectedCallback(onBeatDetected);
      pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);
      maxReady = true;
      Serial.println("MAX30100 initialized!");
    } else {
      Serial.printf("MAX30100 attempt %d failed\n", attempt + 1);
      delay(500);
    }
  }
  if (!maxReady) Serial.println("MAX30100 not found — continuing without SpO2");
  delay(200);

  // TFT
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(2, 2);
  tft.println("ECG Monitor - LIVE Lead I");
  tft.drawFastHLine(0, 12, 160, ST77XX_BLUE);
  tft.setCursor(2, 20);
  tft.setTextColor(maxReady ? ST77XX_GREEN : ST77XX_RED);
  tft.println(maxReady ? "SpO2 sensor OK!" : "SpO2 sensor FAIL");
  delay(500);

  // WiFi
  connectWiFi();

  // TFLite
  tft.setCursor(2, 50);
  tft.setTextColor(ST77XX_YELLOW);
  tft.println("Loading model...");
  tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
  model_tfl    = tflite::GetModel(cnn1d_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interp(model_tfl, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interp = &static_interp;
  interp->AllocateTensors();
  input_tensor  = interp->input(0);
  output_tensor = interp->output(0);

  // Fetch patient
  tft.setCursor(2, 65);
  tft.setTextColor(ST77XX_CYAN);
  tft.println("Fetching patient...");
  fetchPatientId();

  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(2, 2);
  tft.setTextColor(ST77XX_GREEN);
  tft.println("System Ready!");
  tft.drawFastHLine(0, 12, 160, ST77XX_BLUE);
  delay(1000);
  tft.fillScreen(ST77XX_BLACK);

  lastBeatMs = millis();
}

// ── Loop: continuous acquire + infer ──────────────────────────────────────────
void loop() {
  checkMaxHealth();

  tft.fillRect(0, 0, 160, 18, ST77XX_BLACK);
  tft.setCursor(2, 2);
  tft.setTextColor(leadsAttached() ? ST77XX_CYAN : ST77XX_RED);
  tft.setTextSize(1);
  tft.print(leadsAttached() ? "LIVE acquiring..." : "LEAD OFF!");
  if (maxReady) {
    tft.setCursor(95, 2);
    tft.setTextColor(ST77XX_WHITE);
    if (hasReading) tft.printf("%dbpm %d%%", currentHR, currentSpo2);
    else            tft.print("-- --");
  }

  bool windowOk = acquireLiveWindow();   // ~10 s, fills liveWindow[] and draws the trace

  if (!windowOk) {
    tft.fillRect(0, 112, 160, 16, ST77XX_BLACK);
    tft.setCursor(2, 114);
    tft.setTextColor(ST77XX_RED);
    tft.print("Signal lost - retrying");
    delay(500);   // brief pause so the message is readable before the next capture starts
    return;       // skip inference on this window, loop() will be called again immediately
  }

  runLiveInference();    // z-normalize, quantize, infer, display, upload
}
