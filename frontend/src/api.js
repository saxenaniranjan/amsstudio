import axios from "axios";

const RAW_API_BASE = (import.meta.env.VITE_API_URL || "").trim();
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

function apiUrl(path) {
  if (!API_BASE) {
    return path;
  }
  if (API_BASE.endsWith("/api") && path.startsWith("/api/")) {
    return `${API_BASE}${path.slice(4)}`;
  }
  return `${API_BASE}${path}`;
}

const api = axios.create({
  timeout: 120000,
});

export async function previewColumns(files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const { data } = await api.post(apiUrl("/api/columns/preview"), form, {
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

  const { data } = await api.post(apiUrl("/api/sessions/process"), form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchOverview(sessionId, payload) {
  const { data } = await api.post(apiUrl(`/api/sessions/${sessionId}/overview`), payload);
  return data;
}

export async function fetchGraph(sessionId, payload) {
  const { data } = await api.post(apiUrl(`/api/sessions/${sessionId}/graphs`), payload);
  return data;
}

export async function fetchWordCloud(sessionId, payload) {
  const { data } = await api.post(apiUrl(`/api/sessions/${sessionId}/word-cloud`), payload);
  return data;
}

export async function fetchComposite(sessionId, payload) {
  const { data } = await api.post(apiUrl(`/api/sessions/${sessionId}/composite`), payload);
  return data;
}

export async function runQuery(sessionId, payload) {
  const { data } = await api.post(apiUrl(`/api/sessions/${sessionId}/query`), payload);
  return data;
}

export function summaryExportUrl(sessionId) {
  return apiUrl(`/api/sessions/${sessionId}/export/summary`);
}

export function enrichedCsvUrl(sessionId) {
  return apiUrl(`/api/sessions/${sessionId}/export/enriched.csv`);
}

export default api;
