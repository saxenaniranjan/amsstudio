import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export async function previewColumns(files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const { data } = await api.post("/api/columns/preview", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function processSession({ files, workspaceName, userMapping, slaThresholdHours }) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("workspace_name", workspaceName);
  form.append("user_mapping", JSON.stringify(userMapping || {}));
  form.append("sla_threshold_hours", JSON.stringify(slaThresholdHours || {}));

  const { data } = await api.post("/api/sessions/process", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchOverview(sessionId, payload) {
  const { data } = await api.post(`/api/sessions/${sessionId}/overview`, payload);
  return data;
}

export async function fetchGraph(sessionId, payload) {
  const { data } = await api.post(`/api/sessions/${sessionId}/graphs`, payload);
  return data;
}

export async function fetchWordCloud(sessionId, payload) {
  const { data } = await api.post(`/api/sessions/${sessionId}/word-cloud`, payload);
  return data;
}

export async function fetchComposite(sessionId, payload) {
  const { data } = await api.post(`/api/sessions/${sessionId}/composite`, payload);
  return data;
}

export async function runQuery(sessionId, payload) {
  const { data } = await api.post(`/api/sessions/${sessionId}/query`, payload);
  return data;
}

export function summaryExportUrl(sessionId) {
  return `${API_BASE}/api/sessions/${sessionId}/export/summary`;
}

export function enrichedCsvUrl(sessionId) {
  return `${API_BASE}/api/sessions/${sessionId}/export/enriched.csv`;
}

export default api;
