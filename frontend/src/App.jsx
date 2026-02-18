import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import Plot from "react-plotly.js";

import {
  enrichedCsvUrl,
  fetchComposite,
  fetchGraph,
  fetchOverview,
  fetchWordCloud,
  previewColumns,
  processSession,
  runQuery,
  summaryExportUrl,
} from "./api";

const LOGIN_EMAIL = "niranjan.saxena@coforge.com";
const LOGIN_PASSWORD = "test123";
const STORAGE_KEY = "ams_react_state_v2";

const REQUIRED_MAPPING = [
  { key: "ticket_id", label: "Incident Number" },
  { key: "team", label: "Assignment Group" },
  { key: "service", label: "Application" },
];

const DEFAULT_SLA = { P1: 4, P2: 8, P3: 24, P4: 72 };
const DEFAULT_OUTPUT_SETTINGS = {
  insight: true,
  graph: true,
  recommendations: true,
  references: true,
};

const INDUSTRY_OPTIONS = [
  "Banking & Financial Services",
  "Insurance",
  "Travel, Transportation & Hospitality",
  "Healthcare & Life Sciences",
  "Retail & Consumer Goods",
  "Public Sector",
  "Energy & Utilities",
];

const TOOL_CATALOG = [
  {
    id: "ticketx",
    title: "Ticket-X Analyzer",
    description:
      "Analyze ticket dumps for categorization, trend detection, recurring issue analysis, and workload forecasting.",
  },
  {
    id: "ams_agents",
    title: "AMS Agents",
    description:
      "Collection of AI agents for semantic search, predictive analytics, and self-healing recommendations.",
  },
  {
    id: "transition_studio",
    title: "Transition Studio",
    description:
      "AI-assisted transition planning with governance, risk and milestone management capabilities.",
  },
  {
    id: "ams_maturity_index",
    title: "AMS Maturity Index",
    description:
      "Framework to assess engagement maturity and guide optimization initiatives across AMS landscapes.",
  },
];

const INTEGRATIONS = [
  {
    id: "servicenow",
    name: "ServiceNow",
    description: "IT service management and workflow automation platform",
    url: "https://www.servicenow.com",
  },
  {
    id: "confluence",
    name: "Confluence",
    description: "Collaboration and documentation tool by Atlassian",
    url: "https://www.atlassian.com/software/confluence",
  },
  {
    id: "sharepoint",
    name: "SharePoint",
    description: "Microsoft platform for document management and collaboration",
    url: "https://www.microsoft.com/microsoft-365/sharepoint/collaboration",
  },
  {
    id: "bmc",
    name: "BMC",
    description: "Enterprise IT service and operations management solution",
    url: "https://www.bmc.com",
  },
  {
    id: "pagerduty",
    name: "PagerDuty",
    description: "Incident response and alerting platform for IT teams",
    url: "https://www.pagerduty.com",
  },
  {
    id: "dynatrace",
    name: "Dynatrace",
    description: "Application performance monitoring and observability tool",
    url: "https://www.dynatrace.com",
  },
  {
    id: "teams",
    name: "Teams",
    description: "Microsoft collaboration platform for communication workflows",
    url: "https://www.microsoft.com/microsoft-teams/group-chat-software",
  },
  {
    id: "slack",
    name: "Slack",
    description: "Messaging and collaboration platform for teams",
    url: "https://slack.com",
  },
];

const DASHBOARD_GRAPHS = [
  { id: "graph_1", title: "Inflow, Outflow and Backlog", category: "Incident Volumetrics" },
  { id: "graph_2", title: "SLA Breach Trend", category: "Delivery Compliance" },
  { id: "graph_3", title: "Top Issue Categories", category: "Incident Composition" },
  { id: "graph_4", title: "Top MTTR by Assignment Group", category: "Efficiency" },
  { id: "graph_5", title: "Top Recurring Issue Patterns", category: "Incident Composition" },
  { id: "graph_6", title: "Team Performance Over Time", category: "Performance" },
  { id: "graph_7", title: "Top Aged Open Tickets", category: "Incident Volumetrics" },
  { id: "graph_8", title: "Time Trend Heatmap", category: "Performance & Workload" },
];

function nowIso() {
  return new Date().toISOString();
}

function buildEmptyTicketX() {
  return {
    sessionId: "",
    columns: [],
    mapping: {},
    mappingSuggestions: {},
    sla: { ...DEFAULT_SLA },
    overview: null,
    selectedFilters: {},
    dateRange: { start: "", end: "" },
    uploadHistory: [],
    conversation: [],
    outputSettings: { ...DEFAULT_OUTPUT_SETTINGS },
  };
}

function makeWorkspace(seed) {
  return {
    id: seed.id,
    name: seed.name,
    industry: seed.industry,
    description: seed.description,
    tools: seed.tools,
    createdAt: seed.createdAt || nowIso(),
    updatedAt: seed.updatedAt || nowIso(),
    teamMembers: seed.teamMembers || [
      {
        name: "Niranjan",
        role: "admin",
        email: LOGIN_EMAIL,
      },
    ],
    fileStorage: seed.fileStorage || [],
    integrations: seed.integrations || {},
    ticketx: { ...buildEmptyTicketX(), ...(seed.ticketx || {}) },
  };
}

function defaultState() {
  return {
    auth: {
      isAuthenticated: false,
      email: "",
      name: "Niranjan",
    },
    workspaces: [
      makeWorkspace({
        id: "ws_test",
        name: "Test",
        industry: "Insurance",
        description: "Test workspace for Ticket-X analysis and operations monitoring.",
        tools: ["ticketx"],
        createdAt: nowIso(),
      }),
      makeWorkspace({
        id: "ws_adecco",
        name: "Adecco",
        industry: "Public Sector",
        description: "HR and workforce services operations workspace.",
        tools: ["ticketx"],
        createdAt: new Date(Date.now() - 5 * 24 * 3600 * 1000).toISOString(),
      }),
      makeWorkspace({
        id: "ws_virgin",
        name: "Virgin Voyages",
        industry: "Travel, Transportation & Hospitality",
        description: "Cruise operations and guest services analytics workspace.",
        tools: ["ticketx"],
        createdAt: new Date(Date.now() - 32 * 24 * 3600 * 1000).toISOString(),
      }),
    ],
  };
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return defaultState();
    }
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.workspaces)) {
      return defaultState();
    }
    return {
      ...defaultState(),
      ...parsed,
      workspaces: parsed.workspaces.map((item) => makeWorkspace(item)),
    };
  } catch (_error) {
    return defaultState();
  }
}

function workspaceSummaryDate(iso) {
  if (!iso) {
    return "unknown";
  }
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / (24 * 3600 * 1000));
  if (days <= 0) return "Created seconds ago";
  if (days === 1) return "Created 1 day ago";
  if (days < 30) return `Created ${days} days ago`;
  const months = Math.floor(days / 30);
  return `Created ${months} month${months > 1 ? "s" : ""} ago`;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Math.round(Number(value)).toLocaleString();
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(digits)}%`;
}

function toLabel(key) {
  return String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function normalizeFiltersFromOverview(overview) {
  const filters = {};
  const columns = overview?.filter_columns || [];
  columns.forEach((col) => {
    filters[col] = [...(overview?.filter_values?.[col] || [])];
  });
  return filters;
}

const AppStateContext = createContext(null);

function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error("AppStateContext missing");
  }
  return context;
}

function AppStateProvider({ children }) {
  const [state, setState] = useState(loadState);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  const api = useMemo(
    () => ({
      login(email, password) {
        const normalized = String(email || "").trim().toLowerCase();
        const ok = normalized === LOGIN_EMAIL && password === LOGIN_PASSWORD;
        if (!ok) {
          return { ok: false, message: "Invalid credentials. Use the credentials provided in the PRD walkthrough." };
        }
        setState((prev) => ({
          ...prev,
          auth: {
            isAuthenticated: true,
            email: normalized,
            name: "Niranjan",
          },
        }));
        return { ok: true };
      },
      logout() {
        setState((prev) => ({
          ...prev,
          auth: { isAuthenticated: false, email: "", name: "Niranjan" },
        }));
      },
      createWorkspace(payload) {
        const id = `ws_${Math.random().toString(36).slice(2, 9)}`;
        const now = nowIso();
        const record = makeWorkspace({
          ...payload,
          id,
          createdAt: now,
          updatedAt: now,
          ticketx: buildEmptyTicketX(),
        });
        setState((prev) => ({ ...prev, workspaces: [record, ...prev.workspaces] }));
        return record;
      },
      updateWorkspace(workspaceId, payload) {
        setState((prev) => ({
          ...prev,
          workspaces: prev.workspaces.map((ws) =>
            ws.id === workspaceId
              ? {
                  ...ws,
                  ...payload,
                  updatedAt: nowIso(),
                }
              : ws
          ),
        }));
      },
      patchTicketX(workspaceId, patch) {
        setState((prev) => {
          let changed = false;
          const updated = prev.workspaces.map((ws) => {
            if (ws.id !== workspaceId) return ws;
            const nextTicketx = {
              ...buildEmptyTicketX(),
              ...ws.ticketx,
              ...patch,
            };
            const same = JSON.stringify(nextTicketx) === JSON.stringify(ws.ticketx || {});
            if (same) {
              return ws;
            }
            changed = true;
            return {
              ...ws,
              updatedAt: nowIso(),
              ticketx: nextTicketx,
            };
          });
          return changed ? { ...prev, workspaces: updated } : prev;
        });
      },
      addUsers(workspaceId, users) {
        setState((prev) => ({
          ...prev,
          workspaces: prev.workspaces.map((ws) =>
            ws.id === workspaceId
              ? {
                  ...ws,
                  updatedAt: nowIso(),
                  teamMembers: [...ws.teamMembers, ...users],
                }
              : ws
          ),
        }));
      },
      addFileEntries(workspaceId, entries) {
        setState((prev) => ({
          ...prev,
          workspaces: prev.workspaces.map((ws) =>
            ws.id === workspaceId
              ? {
                  ...ws,
                  updatedAt: nowIso(),
                  fileStorage: [...entries, ...ws.fileStorage],
                }
              : ws
          ),
        }));
      },
      toggleIntegration(workspaceId, integrationId) {
        setState((prev) => ({
          ...prev,
          workspaces: prev.workspaces.map((ws) => {
            if (ws.id !== workspaceId) return ws;
            return {
              ...ws,
              updatedAt: nowIso(),
              integrations: {
                ...ws.integrations,
                [integrationId]: !ws.integrations?.[integrationId],
              },
            };
          }),
        }));
      },
      getWorkspace(workspaceId) {
        return state.workspaces.find((ws) => ws.id === workspaceId) || null;
      },
    }),
    [state.workspaces]
  );

  return <AppStateContext.Provider value={{ state, api }}>{children}</AppStateContext.Provider>;
}

function RequireAuth({ children }) {
  const {
    state: { auth },
  } = useAppState();
  if (!auth.isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function AppShell({ children, pageActions = null }) {
  const {
    state: { auth, workspaces },
    api,
  } = useAppState();
  const location = useLocation();

  const firstWorkspaceId = workspaces[0]?.id;
  const ticketXPath = firstWorkspaceId ? `/chat-bot/${firstWorkspaceId}` : "/workspaces";

  return (
    <div className="ams-shell">
      <aside className="ams-rail">
        <div className="rail-logo">\u2261</div>
        <Link className={location.pathname.startsWith("/workspace") || location.pathname.startsWith("/workspaces") ? "rail-item active" : "rail-item"} to="/workspaces" title="Workspaces">
          \u25a3
        </Link>
        <Link className={location.pathname.startsWith("/chat-bot") ? "rail-item active" : "rail-item"} to={ticketXPath} title="Ticket-X">
          \u25ce
        </Link>
      </aside>

      <div className="ams-main">
        <header className="ams-topbar">
          <Link className="brand" to="/workspaces">
            <span className="brand-dot">AMS</span>
            <span className="brand-studio">STUDIO</span>
          </Link>
          <div className="topbar-right">
            {pageActions}
            <span className="user-chip">{auth.name}</span>
            <button className="ghost-btn" onClick={() => api.logout()}>
              Logout
            </button>
          </div>
        </header>
        <main className="ams-content">{children}</main>
      </div>
    </div>
  );
}

function LoginPage() {
  const navigate = useNavigate();
  const { api } = useAppState();

  const [email, setEmail] = useState(LOGIN_EMAIL);
  const [password, setPassword] = useState(LOGIN_PASSWORD);
  const [error, setError] = useState("");

  function onSubmit(event) {
    event.preventDefault();
    const result = api.login(email, password);
    if (!result.ok) {
      setError(result.message || "Login failed.");
      return;
    }
    navigate("/workspaces", { replace: true });
  }

  return (
    <div className="login-page">
      <div className="login-right">
        <h1 className="login-title">
          Welcome to <span>AMS Studio</span>
        </h1>
        <form className="login-card" onSubmit={onSubmit}>
          <h2>Sign in to your account</h2>
          <p>Access your AI powered workspace and insights.</p>

          <label>
            Email
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>

          <label>
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>

          {error ? <div className="error-box">{error}</div> : null}

          <button type="submit" className="primary-btn">
            Login
          </button>
          <button type="button" className="outline-btn">
            Login using SSO (Single Sign on)
          </button>
        </form>
        <div className="login-footer">2026 \u00a9 Coforge, All rights reserved.</div>
      </div>
    </div>
  );
}

function WorkspacesPage() {
  const navigate = useNavigate();
  const {
    state: { workspaces },
  } = useAppState();

  const [viewMode, setViewMode] = useState("grid");

  return (
    <AppShell>
      <section className="page-head">
        <div>
          <h1>Your Workspaces ({workspaces.length})</h1>
          <p>Manage, monitor and explore your configured AMS workspaces.</p>
        </div>
        <div className="inline-actions">
          <button className={viewMode === "grid" ? "chip-btn active" : "chip-btn"} onClick={() => setViewMode("grid")}>
            Grid
          </button>
          <button className={viewMode === "list" ? "chip-btn active" : "chip-btn"} onClick={() => setViewMode("list")}>
            List
          </button>
          <button className="primary-btn" onClick={() => navigate("/workspace-create")}>
            + New Workspace
          </button>
        </div>
      </section>

      <div className={viewMode === "grid" ? "workspace-grid" : "workspace-list"}>
        {workspaces.map((workspace) => (
          <button key={workspace.id} className="workspace-card" onClick={() => navigate(`/workspace-detail/${workspace.id}`)}>
            <div className="workspace-card-row">
              <h3>{workspace.name}</h3>
              <span className="muted">\u22ee</span>
            </div>
            <p className="muted">Industry: {workspace.industry}</p>
            <p>{workspace.description}</p>
            <div className="workspace-footer">
              <span>Ticket-X Analyzer</span>
              <span>{workspaceSummaryDate(workspace.createdAt)}</span>
            </div>
          </button>
        ))}
      </div>

      <section className="catalog-block">
        <h2>Intelligent Tools You Can Add to Your Workspace</h2>
        <p>
          AMS Studio offers a unified platform for ticket analytics, root cause detection, and automated recommendations.
          Modules are powered by AI to deliver actionable insights.
        </p>

        <div className="tool-grid">
          {TOOL_CATALOG.map((tool) => (
            <article key={tool.id} className="tool-card">
              <div className="tool-icon">\u25cc</div>
              <h3>{tool.title}</h3>
              <p>{tool.description}</p>
              <span className="linkish">Learn More \u2192</span>
            </article>
          ))}
        </div>
      </section>
    </AppShell>
  );
}

function WorkspaceFormPage() {
  const navigate = useNavigate();
  const { workspaceId } = useParams();
  const { api } = useAppState();

  const editing = Boolean(workspaceId);
  const workspace = editing ? api.getWorkspace(workspaceId) : null;

  const [name, setName] = useState(workspace?.name || "");
  const [industry, setIndustry] = useState(workspace?.industry || "");
  const [description, setDescription] = useState(workspace?.description || "");
  const [tools, setTools] = useState(workspace?.tools || []);
  const [error, setError] = useState("");

  if (editing && !workspace) {
    return <Navigate to="/workspaces" replace />;
  }

  function toggleTool(toolId) {
    setTools((prev) => (prev.includes(toolId) ? prev.filter((id) => id !== toolId) : [...prev, toolId]));
  }

  function onSave() {
    if (!name.trim() || !industry || !description.trim() || tools.length === 0) {
      setError("Please fill all mandatory fields and choose at least one tool.");
      return;
    }

    const payload = {
      name: name.trim(),
      industry,
      description: description.trim(),
      tools,
    };

    if (editing) {
      api.updateWorkspace(workspaceId, payload);
      navigate(`/workspace-detail/${workspaceId}`);
      return;
    }

    const created = api.createWorkspace(payload);
    navigate(`/workspace-detail/${created.id}`);
  }

  return (
    <AppShell>
      <section className="page-head">
        <h1>{editing ? "Edit Workspace" : "Create New Workspace"}</h1>
        <div className="inline-actions">
          <button className="chip-btn" onClick={() => navigate(-1)}>
            Cancel
          </button>
          <button className="primary-btn" onClick={onSave}>
            {editing ? "Update Workspace" : "Save Workspace"}
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="form-row">
          <label>
            Name *
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Workspace Name" />
          </label>
          <label>
            Industry *
            <select value={industry} onChange={(e) => setIndustry(e.target.value)}>
              <option value="">Select Industry</option>
              {INDUSTRY_OPTIONS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label>
          Description *
          <textarea
            rows={4}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Keep your description brief and specific."
          />
        </label>

        {error ? <div className="error-box">{error}</div> : null}

        <h2>Select Tools ({tools.length}) *</h2>
        <div className="tool-grid">
          {TOOL_CATALOG.map((tool) => {
            const selected = tools.includes(tool.id);
            return (
              <article key={tool.id} className={selected ? "tool-card selected" : "tool-card"}>
                <div className="tool-icon">\u25cc</div>
                <h3>{tool.title}</h3>
                <p>{tool.description}</p>
                <div className="tool-actions">
                  <span className="linkish">Learn More \u2192</span>
                  <button className={selected ? "chip-btn active" : "chip-btn"} onClick={() => toggleTool(tool.id)}>
                    {selected ? "Remove" : "+ Add to Workspace"}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </AppShell>
  );
}

function WorkspaceDetailPage() {
  const navigate = useNavigate();
  const { workspaceId } = useParams();
  const { api } = useAppState();

  const workspace = api.getWorkspace(workspaceId);
  if (!workspace) {
    return <Navigate to="/workspaces" replace />;
  }

  return (
    <AppShell>
      <section className="page-head">
        <div>
          <h1>{workspace.name}</h1>
          <p>{workspace.description}</p>
        </div>
        <div className="inline-actions">
          <button className="link-btn" onClick={() => navigate(`/workspace-edit/${workspace.id}`)}>
            Edit Workspace
          </button>
          <button className="chip-btn" onClick={() => navigate(`/workspace-settings/${workspace.id}`)}>
            Settings
          </button>
        </div>
      </section>

      <section className="summary-grid">
        <article>
          <h3>{workspace.name}</h3>
          <p>Workspace</p>
        </article>
        <article>
          <h3>{workspace.tools.length}</h3>
          <p>No. of tools</p>
        </article>
        <article>
          <h3>{workspace.industry}</h3>
          <p>Industry</p>
        </article>
        <article>
          <h3>{workspace.teamMembers.length}</h3>
          <p>Team members</p>
        </article>
      </section>

      <section>
        <h2>Your Workspace Tools</h2>
        <div className="tool-grid">
          {workspace.tools.map((toolId) => {
            const tool = TOOL_CATALOG.find((item) => item.id === toolId);
            if (!tool) return null;
            const ticketX = toolId === "ticketx";
            return (
              <article key={tool.id} className="tool-card">
                <div className="tool-icon">\u25cc</div>
                <h3>{tool.title}</h3>
                <p>{tool.description}</p>
                <div className="tool-actions">
                  <span className="linkish">Learn More \u2192</span>
                  <button
                    className={ticketX ? "primary-btn" : "chip-btn"}
                    onClick={() => ticketX && navigate(`/chat-bot/${workspace.id}`)}
                    disabled={!ticketX}
                  >
                    {ticketX ? "Launch" : "Coming Soon"}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </AppShell>
  );
}

function WorkspaceSettingsPage() {
  const { workspaceId } = useParams();
  const { api } = useAppState();
  const workspace = api.getWorkspace(workspaceId);

  const [activeTab, setActiveTab] = useState("users");
  const [newUser, setNewUser] = useState({ name: "", role: "member", email: "" });

  if (!workspace) {
    return <Navigate to="/workspaces" replace />;
  }

  function addUser() {
    if (!newUser.name.trim() || !newUser.email.trim()) {
      return;
    }
    api.addUsers(workspace.id, [{ ...newUser }]);
    setNewUser({ name: "", role: "member", email: "" });
  }

  return (
    <AppShell>
      <section className="page-head">
        <div>
          <h1>Settings</h1>
          <p>Manage, update settings</p>
        </div>
      </section>

      <div className="tabs-row">
        <button className={activeTab === "users" ? "tab-btn active" : "tab-btn"} onClick={() => setActiveTab("users")}>
          User Access
        </button>
        <button className={activeTab === "files" ? "tab-btn active" : "tab-btn"} onClick={() => setActiveTab("files")}>
          File Storage
        </button>
        <button
          className={activeTab === "integrations" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("integrations")}
        >
          Integrations
        </button>
      </div>

      {activeTab === "users" && (
        <section className="panel">
          <div className="section-head">
            <div>
              <h2>Manage team ({workspace.teamMembers.length})</h2>
              <p>Manage team members, their roles and access</p>
            </div>
          </div>

          <table className="plain-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Current Role</th>
                <th>Email address</th>
              </tr>
            </thead>
            <tbody>
              {workspace.teamMembers.map((member, index) => (
                <tr key={`${member.email}_${index}`}>
                  <td>{member.name}</td>
                  <td>{member.role}</td>
                  <td>{member.email}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="form-row compact">
            <label>
              Name
              <input value={newUser.name} onChange={(e) => setNewUser((prev) => ({ ...prev, name: e.target.value }))} />
            </label>
            <label>
              Role
              <select value={newUser.role} onChange={(e) => setNewUser((prev) => ({ ...prev, role: e.target.value }))}>
                <option value="admin">admin</option>
                <option value="member">member</option>
              </select>
            </label>
            <label>
              Email
              <input
                type="email"
                value={newUser.email}
                onChange={(e) => setNewUser((prev) => ({ ...prev, email: e.target.value }))}
              />
            </label>
            <button className="primary-btn" onClick={addUser}>
              Add User
            </button>
          </div>
        </section>
      )}

      {activeTab === "files" && (
        <section className="panel">
          <h2>Workspace Files</h2>
          {workspace.fileStorage.length === 0 ? (
            <div className="empty-box">No files yet. Process a session in Ticket-X to populate storage history.</div>
          ) : (
            <table className="plain-table">
              <thead>
                <tr>
                  <th>File Name</th>
                  <th>Rows</th>
                  <th>Uploaded At</th>
                </tr>
              </thead>
              <tbody>
                {workspace.fileStorage.map((item, index) => (
                  <tr key={`${item.file_name}_${index}`}>
                    <td>{item.file_name}</td>
                    <td>{item.rows ?? "-"}</td>
                    <td>{new Date(item.uploaded_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {activeTab === "integrations" && (
        <section className="integrations-list">
          {INTEGRATIONS.map((item) => {
            const connected = Boolean(workspace.integrations?.[item.id]);
            return (
              <article key={item.id} className="integration-card">
                <div>
                  <h3>{item.name}</h3>
                  <p>{item.description}</p>
                  <a href={item.url} target="_blank" rel="noreferrer" className="linkish">
                    Learn More
                  </a>
                </div>
                <button className={connected ? "chip-btn active" : "primary-btn"} onClick={() => api.toggleIntegration(workspace.id, item.id)}>
                  {connected ? "Connected" : "Connect"}
                </button>
              </article>
            );
          })}
        </section>
      )}
    </AppShell>
  );
}

function DataTable({ rows }) {
  if (!rows || rows.length === 0) {
    return <div className="empty-box">No rows available.</div>;
  }
  const headers = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table className="plain-table">
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{toLabel(header)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 120).map((row, index) => (
            <tr key={`row_${index}`}>
              {headers.map((header) => (
                <td key={`${header}_${index}`}>{String(row[header] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PlotPanel({ figure, height = 280 }) {
  if (!figure) {
    return <div className="empty-box">Chart not loaded.</div>;
  }

  return (
    <Plot
      data={figure.data || []}
      layout={{
        ...figure.layout,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        margin: { l: 40, r: 20, t: 40, b: 40 },
      }}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: `${height}px` }}
      useResizeHandler
    />
  );
}

function OutputBox({ output }) {
  if (!output) {
    return null;
  }

  return (
    <section className="panel">
      {output.kind === "clarification" ? (
        <div className="insight-banner">
          <strong>Clarification Needed:</strong> {output.text}
        </div>
      ) : null}
      {output.summary ? <p className="summary-lede">{output.summary}</p> : null}
      {output.text && output.kind !== "clarification" ? <p className="summary-lede">{output.text}</p> : null}
      {output.agent_trace?.validation ? (
        <div className={output.agent_trace.validation.is_valid ? "insight-banner ok" : "insight-banner"}>
          {output.agent_trace.validation.is_valid
            ? "Validator Agent: Passed"
            : `Validator Agent: ${output.agent_trace.validation.issues.join("; ")}`}
        </div>
      ) : null}

      {output.findings?.length ? (
        <div className="subpanel">
          <h3>Findings</h3>
          <ul className="plain-list">
            {output.findings.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {output.recommendations?.length ? (
        <div className="subpanel">
          <h3>Recommendations</h3>
          <ul className="plain-list">
            {output.recommendations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {output.figure ? <PlotPanel figure={output.figure} /> : null}
      {output.data?.length ? <DataTable rows={output.data} /> : null}
    </section>
  );
}

function GlobalFilterModal({ open, onClose, overview, selectedFilters, setSelectedFilters, dateRange, setDateRange, onApply }) {
  if (!open) {
    return null;
  }

  const filterColumns = overview?.filter_columns || [];
  const filterValues = overview?.filter_values || {};

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="section-head">
          <h3>Global Filters</h3>
          <button className="ghost-btn" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="form-row compact">
          <label>
            Start Date
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange((prev) => ({ ...prev, start: e.target.value }))}
            />
          </label>
          <label>
            End Date
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange((prev) => ({ ...prev, end: e.target.value }))}
            />
          </label>
        </div>

        <div className="filter-grid">
          {filterColumns.map((column) => (
            <label key={column}>
              {toLabel(column)}
              <select
                multiple
                value={selectedFilters[column] || []}
                onChange={(e) => {
                  const values = Array.from(e.target.selectedOptions).map((item) => item.value);
                  setSelectedFilters((prev) => ({ ...prev, [column]: values }));
                }}
              >
                {(filterValues[column] || []).map((value) => (
                  <option key={`${column}_${value}`} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>

        <div className="inline-actions right">
          <button className="chip-btn" onClick={onClose}>
            Cancel
          </button>
          <button className="primary-btn" onClick={onApply}>
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}

function ChatBotPage() {
  const { workspaceId } = useParams();
  const navigate = useNavigate();
  const { api } = useAppState();

  const workspace = api.getWorkspace(workspaceId);
  const storedTicketX = workspace?.ticketx || buildEmptyTicketX();

  const [files, setFiles] = useState([]);
  const [columns, setColumns] = useState(storedTicketX.columns || []);
  const [mappingSuggestions, setMappingSuggestions] = useState(storedTicketX.mappingSuggestions || {});
  const [mapping, setMapping] = useState(storedTicketX.mapping || {});
  const [sla, setSla] = useState(storedTicketX.sla || { ...DEFAULT_SLA });

  const [sessionId, setSessionId] = useState(storedTicketX.sessionId || "");
  const [overview, setOverview] = useState(storedTicketX.overview || null);
  const [selectedFilters, setSelectedFilters] = useState(storedTicketX.selectedFilters || {});
  const [dateRange, setDateRange] = useState(storedTicketX.dateRange || { start: "", end: "" });
  const [uploadHistory, setUploadHistory] = useState(storedTicketX.uploadHistory || []);
  const [conversation, setConversation] = useState(storedTicketX.conversation || []);

  const [showDashboard, setShowDashboard] = useState(Boolean(storedTicketX.overview));
  const [showFilters, setShowFilters] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);

  const [graphs, setGraphs] = useState({});
  const [focusedInsight, setFocusedInsight] = useState("");
  const [wordCloudFigure, setWordCloudFigure] = useState(null);
  const [compositeFigure, setCompositeFigure] = useState(null);
  const [compositeForm, setCompositeForm] = useState({ x_col: "", y_col: "", chart_type: "bar", color_col: "" });

  const [query, setQuery] = useState("");
  const [output, setOutput] = useState(null);

  if (!workspace) {
    return <Navigate to="/workspaces" replace />;
  }

  useEffect(() => {
    api.patchTicketX(workspace.id, {
      sessionId,
      columns,
      mappingSuggestions,
      mapping,
      sla,
      overview,
      selectedFilters,
      dateRange,
      uploadHistory,
      conversation,
    });
  }, [
    api,
    workspace.id,
    sessionId,
    columns,
    mappingSuggestions,
    mapping,
    sla,
    overview,
    selectedFilters,
    dateRange,
    uploadHistory,
    conversation,
  ]);

  const filterColumns = overview?.filter_columns || [];
  const filterValues = overview?.filter_values || {};

  const numericColumns = useMemo(() => {
    const rows = overview?.sample_rows || [];
    if (!rows.length) {
      return ["mttr_hours", "ticket_age_days", "team_performance_index", "sla_threshold_hours"];
    }
    const sample = rows[0];
    return Object.keys(sample).filter((key) => typeof sample[key] === "number");
  }, [overview]);

  function mappingToApiShape() {
    const payload = {};
    Object.entries(mapping).forEach(([canonical, source]) => {
      if (source) {
        payload[source] = canonical;
      }
    });
    return payload;
  }

  function buildFilterPayload() {
    return {
      filters: selectedFilters,
      start_date: dateRange.start || null,
      end_date: dateRange.end || null,
    };
  }

  function handleSessionNotFound(error) {
    const detail = error?.response?.data?.detail || error?.message || "";
    if (!String(detail).toLowerCase().includes("session not found")) {
      return false;
    }
    setSessionId("");
    setOverview(null);
    setGraphs({});
    setOutput(null);
    setShowDashboard(false);
    setStatus("Session expired or server restarted. Please process your file again.");
    return true;
  }

  async function previewMapping() {
    if (!files.length) {
      setStatus("Upload one or more Excel/CSV files first.");
      return;
    }

    setLoading(true);
    setStatus("Reading columns and generating mapping suggestions...");
    try {
      const data = await previewColumns(files);
      const nextColumns = data.columns || [];
      const suggestions = data.mapping_suggestions || {};
      const autoMap = {};
      REQUIRED_MAPPING.forEach((item) => {
        if (suggestions[item.key]?.length) {
          autoMap[item.key] = suggestions[item.key][0];
        }
      });

      setColumns(nextColumns);
      setMappingSuggestions(suggestions);
      setMapping((prev) => ({ ...autoMap, ...prev }));
      const fileErrors = data.file_errors || [];
      if (fileErrors.length) {
        const warning = fileErrors
          .slice(0, 2)
          .map((item) => `${item.file_name}: ${item.error}`)
          .join(" | ");
        setStatus(`Column preview ready with warnings: ${warning}`);
      } else {
        setStatus("Column preview ready.");
      }
    } catch (error) {
      const detail = error?.response?.data?.detail;
      setStatus(typeof detail === "string" ? detail : "Failed to read columns.");
    } finally {
      setLoading(false);
    }
  }

  async function processWorkspace() {
    if (!files.length) {
      setStatus("Upload one or more data files first.");
      return;
    }

    setLoading(true);
    setStatus("Running preprocessing, feature derivation and KPI generation...");

    try {
      const response = await processSession({
        files,
        workspaceName: workspace.name,
        userMapping: mappingToApiShape(),
        slaThresholdHours: sla,
      });

      const nextSessionId = response.session_id;
      const nextOverview = { ...response.overview, columns: response.columns };
      const nextFilters = normalizeFiltersFromOverview(nextOverview);
      const min = response.overview?.date_range?.min?.slice(0, 10) || "";
      const max = response.overview?.date_range?.max?.slice(0, 10) || "";
      const fileEntries = (response.upload_history || []).map((item) => ({
        ...item,
        uploaded_at: item.uploaded_at || nowIso(),
      }));

      setSessionId(nextSessionId);
      setOverview(nextOverview);
      setSelectedFilters(nextFilters);
      setDateRange({ start: min, end: max });
      setUploadHistory(fileEntries);
      setGraphs({});
      setShowDashboard(true);
      const fileErrors = response.file_errors || [];
      const sessionWarning = response.session_warning;
      if (fileErrors.length || sessionWarning) {
        const warning = fileErrors
          .slice(0, 2)
          .map((item) => `${item.file_name}: ${item.error}`)
          .join(" | ");
        const warnings = [warning, sessionWarning].filter(Boolean).join(" | ");
        setStatus(`Workspace processed with warnings: ${warnings}`);
      } else {
        setStatus("Workspace processed successfully. Dashboard is ready.");
      }

      api.addFileEntries(workspace.id, fileEntries);
    } catch (error) {
      const detail = error?.response?.data?.detail;
      setStatus(typeof detail === "string" ? detail : "Failed to process workspace.");
    } finally {
      setLoading(false);
    }
  }

  async function refreshOverviewAndGraphs() {
    if (!sessionId) {
      return;
    }

    setLoading(true);
    setStatus("Applying filters and refreshing dashboard...");

    try {
      const data = await fetchOverview(sessionId, buildFilterPayload());
      const merged = { ...data, columns: overview?.columns || [] };
      setOverview(merged);
      await loadAllGraphs(sessionId, buildFilterPayload());
      setStatus("Dashboard updated with selected filters.");
    } catch (error) {
      if (handleSessionNotFound(error)) {
        return;
      }
      setStatus(error?.response?.data?.detail || "Unable to refresh dashboard.");
    } finally {
      setLoading(false);
      setShowFilters(false);
    }
  }

  async function loadAllGraphs(activeSession, filterPayload) {
    const targetSession = activeSession || sessionId;
    if (!targetSession) {
      return;
    }

    const payload = filterPayload || buildFilterPayload();
    const outputById = {};
    let sessionMissing = false;

    await Promise.all(
      DASHBOARD_GRAPHS.map(async (graphMeta) => {
        try {
          const graph = await fetchGraph(targetSession, { graph_id: graphMeta.id, ...payload });
          outputById[graphMeta.id] = graph;
        } catch (error) {
          const detail = error?.response?.data?.detail || "";
          if (String(detail).toLowerCase().includes("session not found")) {
            sessionMissing = true;
          }
          outputById[graphMeta.id] = null;
        }
      })
    );

    if (sessionMissing) {
      throw new Error("Session not found");
    }

    setGraphs(outputById);
  }

  useEffect(() => {
    if (showDashboard && sessionId && Object.keys(graphs).length === 0) {
      setLoading(true);
      setStatus("Loading PRD graph catalog...");
      loadAllGraphs().finally(() => {
        setLoading(false);
        setStatus("Dashboard loaded.");
      });
    }
  }, [showDashboard, sessionId, graphs]);

  async function runAutonomous(label) {
    if (!sessionId) {
      setStatus("Process workspace first.");
      return;
    }

    setLoading(true);
    setStatus(`${label} in progress...`);

    try {
      const data = await runQuery(sessionId, {
        mode: "autonomous",
        query: "",
        ...buildFilterPayload(),
      });
      setOutput(data);
      setConversation((prev) => [{ mode: label, text: data.summary || "", time: nowIso() }, ...prev]);
      setStatus("Autonomous analysis complete.");
    } catch (error) {
      if (handleSessionNotFound(error)) {
        return;
      }
      setStatus(error?.response?.data?.detail || "Unable to run autonomous analysis.");
    } finally {
      setLoading(false);
    }
  }

  async function runEnablerQuery() {
    if (!sessionId) {
      setStatus("Process workspace first.");
      return;
    }

    setLoading(true);
    setStatus("Running Ticket-X query...");

    try {
      const data = await runQuery(sessionId, {
        mode: "enabler",
        query,
        ...buildFilterPayload(),
      });
      setOutput(data);
      setConversation((prev) => [{ mode: "enabler", text: query, time: nowIso() }, ...prev]);
      setStatus("Query complete.");
    } catch (error) {
      if (handleSessionNotFound(error)) {
        return;
      }
      setStatus(error?.response?.data?.detail || "Unable to process query.");
    } finally {
      setLoading(false);
    }
  }

  async function generateWordCloud() {
    if (!sessionId) {
      setStatus("Process workspace first.");
      return;
    }

    setLoading(true);
    setStatus("Generating word cloud...");

    try {
      const data = await fetchWordCloud(sessionId, buildFilterPayload());
      setWordCloudFigure(data.figure);
      setStatus("Word cloud ready.");
    } catch (error) {
      if (handleSessionNotFound(error)) {
        return;
      }
      setStatus(error?.response?.data?.detail || "Unable to generate word cloud.");
    } finally {
      setLoading(false);
    }
  }

  async function buildComposite() {
    if (!sessionId) {
      setStatus("Process workspace first.");
      return;
    }
    if (!compositeForm.x_col || !compositeForm.y_col) {
      setStatus("Select both X and Y columns for composite graph.");
      return;
    }

    setLoading(true);
    setStatus("Creating composite graph...");

    try {
      const data = await fetchComposite(sessionId, {
        ...buildFilterPayload(),
        ...compositeForm,
        color_col: compositeForm.color_col || null,
      });
      setCompositeFigure(data.figure);
      setStatus("Composite graph ready.");
    } catch (error) {
      if (handleSessionNotFound(error)) {
        return;
      }
      setStatus(error?.response?.data?.detail || "Unable to build composite graph.");
    } finally {
      setLoading(false);
    }
  }

  const cards = overview?.cards || {};
  const report = overview?.report || {};

  const pageActions = (
    <>
      <label className="toggle-wrap">
        <input type="checkbox" checked={showDashboard} onChange={(e) => setShowDashboard(e.target.checked)} />
        <span>Show Dashboard</span>
      </label>
      <button className="chip-btn" onClick={() => navigate(`/chat-bot-settings/${workspace.id}`)}>
        Settings
      </button>
      {sessionId ? (
        <button className="chip-btn" onClick={() => setShowFilters(true)}>
          Global Filters
        </button>
      ) : null}
    </>
  );

  return (
    <AppShell pageActions={pageActions}>
      <section className="page-head">
        <div>
          <h1>{workspace.name}: Ticket-X Analyzer</h1>
          <p>Process ticket dumps, derive PRD metrics, and ask for graph-driven or textual insights.</p>
        </div>
        <div className="status-pill">{loading ? "Working..." : status}</div>
      </section>

      {!sessionId && (
        <section className="panel">
          <h2>Data Ingestion and Mapping</h2>
          <p>
            Upload ITIL exports, map required fields, set SLA thresholds, and create an enriched analytics session.
          </p>

          <div className="form-row compact">
            <label>
              Upload Excel/CSV
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                multiple
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
            </label>
            <button className="chip-btn" onClick={previewMapping}>
              Preview Columns
            </button>
          </div>

          {columns.length > 0 ? (
            <div className="subpanel">
              <h3>Mandatory Mapping</h3>
              <div className="form-row compact">
                {REQUIRED_MAPPING.map((field) => (
                  <label key={field.key}>
                    {field.label}
                    <select
                      value={mapping[field.key] || ""}
                      onChange={(e) => setMapping((prev) => ({ ...prev, [field.key]: e.target.value }))}
                    >
                      <option value="">Auto Detect</option>
                      {(mappingSuggestions[field.key] || []).map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                      {columns
                        .filter((column) => !(mappingSuggestions[field.key] || []).includes(column))
                        .map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                    </select>
                  </label>
                ))}
              </div>

              <h3>SLA Mapping (Hours)</h3>
              <div className="form-row compact">
                {Object.keys(sla).map((key) => (
                  <label key={key}>
                    {key}
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      value={sla[key]}
                      onChange={(e) => setSla((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                    />
                  </label>
                ))}
              </div>

              <button className="primary-btn" onClick={processWorkspace}>
                Process Workspace
              </button>
            </div>
          ) : null}
        </section>
      )}

      {sessionId ? (
        <>
          <section className="mode-grid">
            <button className="mode-card" onClick={() => runAutonomous("summary")}> 
              <h3>Summarize Data for Your Review</h3>
              <p>Build PRD card summary, risks, and recommendations from current filter context.</p>
            </button>
            <button className="mode-card" onClick={() => runAutonomous("auto_detect")}> 
              <h3>Auto-Detect Issues and Generate Insights</h3>
              <p>Scan recurring patterns, SLA risk clusters, and underperforming teams automatically.</p>
            </button>
          </section>

          <section className="panel">
            <label>
              Ask Ticket-X Analyzer Agent
              <textarea
                rows={3}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Example: show graph 2 by priority, top recurring issues for finance, or recommend actions"
              />
            </label>
            <div className="inline-actions">
              <button className="primary-btn" onClick={runEnablerQuery}>
                Run Query
              </button>
              <button className="chip-btn" onClick={() => setQuery("")}>New Chat</button>
              <a className="chip-btn" href={enrichedCsvUrl(sessionId)} target="_blank" rel="noreferrer">
                Export Enriched CSV
              </a>
              <a className="chip-btn" href={summaryExportUrl(sessionId)} target="_blank" rel="noreferrer">
                Export Summary JSON
              </a>
            </div>
          </section>

          <OutputBox output={output} />

          {showDashboard && (
            <section className="dashboard-stack">
              <div className="section-head">
                <h2>Dashboard</h2>
                <button className="chip-btn" onClick={refreshOverviewAndGraphs}>
                  Refresh Dashboard
                </button>
              </div>

              <div className="metrics-strip">
                <article>
                  <h4>Total Tickets</h4>
                  <strong>{formatInt(report?.metrics?.total_tickets)}</strong>
                </article>
                <article>
                  <h4>Open Tickets</h4>
                  <strong>{formatInt(report?.metrics?.open_tickets)}</strong>
                </article>
                <article>
                  <h4>SLA Adherence</h4>
                  <strong>{formatPercent(cards?.delivery_compliance?.sla_adherence_pct)}</strong>
                </article>
                <article>
                  <h4>MTTR (hrs)</h4>
                  <strong>{formatNumber(cards?.efficiency?.mttr_hours)}</strong>
                </article>
                <article>
                  <h4>Recurring Issue %</h4>
                  <strong>{formatPercent(cards?.quality?.recurring_issue_pct)}</strong>
                </article>
                <article>
                  <h4>At Risk Open</h4>
                  <strong>{formatInt(cards?.delivery_compliance?.at_risk_open_tickets)}</strong>
                </article>
              </div>

              <div className="recommend-box">
                <h3>Recommendations</h3>
                <ul className="plain-list">
                  {(report?.recommendations || []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="dashboard-grid">
                {DASHBOARD_GRAPHS.map((graphMeta) => {
                  const graph = graphs[graphMeta.id];
                  return (
                    <article key={graphMeta.id} className="dashboard-card">
                      <div className="dashboard-card-head">
                        <div>
                          <small>{graphMeta.category}</small>
                          <h3>{graphMeta.title}</h3>
                        </div>
                        <button
                          className="link-btn"
                          onClick={() => setFocusedInsight(graph?.insight_hint || "No insight available for this graph.")}
                        >
                          View Insights
                        </button>
                      </div>
                      <PlotPanel figure={graph?.figure || null} height={260} />
                      {graphMeta.id === "graph_7" ? <DataTable rows={graph?.data || []} /> : null}
                    </article>
                  );
                })}
              </div>

              {focusedInsight ? <div className="insight-banner">{focusedInsight}</div> : null}

              <section className="panel">
                <div className="section-head">
                  <h2>Custom Graphs</h2>
                  <button className="chip-btn" onClick={generateWordCloud}>
                    Generate Word Cloud
                  </button>
                </div>

                <PlotPanel figure={wordCloudFigure} height={330} />

                <div className="subpanel">
                  <h3>Create Composite Graph</h3>
                  <div className="form-row compact">
                    <label>
                      X Column
                      <select
                        value={compositeForm.x_col}
                        onChange={(e) => setCompositeForm((prev) => ({ ...prev, x_col: e.target.value }))}
                      >
                        <option value="">Select</option>
                        {(overview?.columns || []).map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Y Column
                      <select
                        value={compositeForm.y_col}
                        onChange={(e) => setCompositeForm((prev) => ({ ...prev, y_col: e.target.value }))}
                      >
                        <option value="">Select</option>
                        {numericColumns.map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Chart Type
                      <select
                        value={compositeForm.chart_type}
                        onChange={(e) => setCompositeForm((prev) => ({ ...prev, chart_type: e.target.value }))}
                      >
                        <option value="bar">Bar</option>
                        <option value="line">Line</option>
                        <option value="scatter">Scatter</option>
                        <option value="histogram">Histogram</option>
                        <option value="box">Box</option>
                      </select>
                    </label>
                    <label>
                      Color Column
                      <select
                        value={compositeForm.color_col}
                        onChange={(e) => setCompositeForm((prev) => ({ ...prev, color_col: e.target.value }))}
                      >
                        <option value="">None</option>
                        {(overview?.columns || []).map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <button className="primary-btn" onClick={buildComposite}>
                    Build Composite Graph
                  </button>
                  <PlotPanel figure={compositeFigure} height={330} />
                </div>
              </section>

              <section className="panel">
                <h2>Conversation History</h2>
                <DataTable rows={conversation} />
              </section>
            </section>
          )}
        </>
      ) : null}

      <GlobalFilterModal
        open={showFilters}
        onClose={() => setShowFilters(false)}
        overview={overview}
        selectedFilters={selectedFilters}
        setSelectedFilters={setSelectedFilters}
        dateRange={dateRange}
        setDateRange={setDateRange}
        onApply={refreshOverviewAndGraphs}
      />
    </AppShell>
  );
}

function ChatBotSettingsPage() {
  const { workspaceId } = useParams();
  const { api } = useAppState();
  const workspace = api.getWorkspace(workspaceId);

  const [settings, setSettings] = useState(workspace?.ticketx?.outputSettings || { ...DEFAULT_OUTPUT_SETTINGS });
  const [saved, setSaved] = useState(false);

  if (!workspace) {
    return <Navigate to="/workspaces" replace />;
  }

  function toggle(field) {
    setSettings((prev) => ({ ...prev, [field]: !prev[field] }));
  }

  function submit() {
    api.patchTicketX(workspace.id, { outputSettings: settings });
    setSaved(true);
    setTimeout(() => setSaved(false), 1800);
  }

  const sampleFigure = {
    data: [
      {
        type: "bar",
        x: ["Jul", "Aug", "Sep", "Oct"],
        y: [15, 16, 11, 14],
        marker: { color: "#3f9fda" },
      },
    ],
    layout: { title: "Sample Preview" },
  };

  return (
    <AppShell>
      <section className="page-head">
        <div>
          <h1>Settings</h1>
          <p>Select the format of output you want the agent to generate.</p>
        </div>
      </section>

      <section className="settings-split">
        <article className="panel">
          <h2>Customization</h2>
          <div className="option-stack">
            <label className="option-row">
              <input type="checkbox" checked={settings.insight} onChange={() => toggle("insight")} />
              <div>
                <h3>Insight</h3>
                <p>Provide short description of problems found while analyzing ticket data.</p>
              </div>
            </label>
            <label className="option-row">
              <input type="checkbox" checked={settings.graph} onChange={() => toggle("graph")} />
              <div>
                <h3>Graph</h3>
                <p>Create chart output showing trend over time or any relevant business dimension.</p>
              </div>
            </label>
            <label className="option-row">
              <input type="checkbox" checked={settings.recommendations} onChange={() => toggle("recommendations")} />
              <div>
                <h3>Recommendations</h3>
                <p>Generate practical actions to improve SLA compliance and delivery quality.</p>
              </div>
            </label>
            <label className="option-row">
              <input type="checkbox" checked={settings.references} onChange={() => toggle("references")} />
              <div>
                <h3>References</h3>
                <p>Include references to relevant tickets or source slices used for the analysis.</p>
              </div>
            </label>
          </div>

          <div className="inline-actions">
            <button className="primary-btn" onClick={submit}>
              Submit
            </button>
            {saved ? <span className="ok-tag">Saved</span> : null}
          </div>
        </article>

        <article className="panel">
          <h2>Sample Preview</h2>
          <p>
            A spike of access requests over the last quarter indicates a recurring operational pattern in low-support
            windows.
          </p>
          <PlotPanel figure={sampleFigure} height={230} />
          <div className="subpanel">
            <h3>Recommendations</h3>
            <p>Implement pre-approved access batches or temporary weekend policies for recurring roles.</p>
          </div>
        </article>
      </section>
    </AppShell>
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/workspaces"
          element={
            <RequireAuth>
              <WorkspacesPage />
            </RequireAuth>
          }
        />
        <Route
          path="/workspace-create"
          element={
            <RequireAuth>
              <WorkspaceFormPage />
            </RequireAuth>
          }
        />
        <Route
          path="/workspace-edit/:workspaceId"
          element={
            <RequireAuth>
              <WorkspaceFormPage />
            </RequireAuth>
          }
        />
        <Route
          path="/workspace-detail/:workspaceId"
          element={
            <RequireAuth>
              <WorkspaceDetailPage />
            </RequireAuth>
          }
        />
        <Route
          path="/workspace-settings/:workspaceId"
          element={
            <RequireAuth>
              <WorkspaceSettingsPage />
            </RequireAuth>
          }
        />
        <Route
          path="/chat-bot/:workspaceId"
          element={
            <RequireAuth>
              <ChatBotPage />
            </RequireAuth>
          }
        />
        <Route
          path="/chat-bot-settings/:workspaceId"
          element={
            <RequireAuth>
              <ChatBotSettingsPage />
            </RequireAuth>
          }
        />
        <Route path="/" element={<Navigate to="/workspaces" replace />} />
        <Route path="*" element={<Navigate to="/workspaces" replace />} />
      </Routes>
    </AppStateProvider>
  );
}
