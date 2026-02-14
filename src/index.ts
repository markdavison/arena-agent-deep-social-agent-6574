import { existsSync, readFileSync } from "node:fs";
import { createMCPClient } from "@ai-sdk/mcp";
import { Experimental_StdioMCPTransport } from "@ai-sdk/mcp/mcp-stdio";
import type { ToolSet } from "ai";
import { runStrategy } from "./strategy.js";
import { runConfigStrategy } from "./config-strategy.js";
import { sanitizeTools } from "./sanitize-tools.js";

interface McpHttpTransport {
  type: "sse" | "http";
  url: string;
  params?: Record<string, string>;
  headers?: Record<string, string>;
}

interface McpStdioTransport {
  type: "stdio";
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

type McpTransport = McpHttpTransport | McpStdioTransport;

interface McpServerConfig {
  requires?: string[];
  transport: McpTransport;
}

interface AgentConfig {
  system_prompt: string;
  model: {
    provider: string;
    model_id: string;
    base_url?: string;
  };
  mcp_servers?: Record<string, McpServerConfig>;
}

interface ConnectedServer {
  name: string;
  client: Awaited<ReturnType<typeof createMCPClient>>;
  tools: ToolSet;
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `Missing required env var: ${name}. ` +
        `Set it in .env or GitHub Secrets.`,
    );
  }
  return value;
}

function expandEnvVars(str: string): string {
  return str.replace(
    /\$\{([^}]+)\}/g,
    (_, name: string) => process.env[name] ?? "",
  );
}

function expandAllEnvVars<T>(value: T): T {
  if (typeof value === "string") return expandEnvVars(value) as T;
  if (Array.isArray(value)) return value.map(expandAllEnvVars) as T;
  if (value !== null && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(
      value as Record<string, unknown>,
    )) {
      result[k] = expandAllEnvVars(v);
    }
    return result as T;
  }
  return value;
}

function buildTransportUrl(
  url: string,
  params?: Record<string, string>,
): string {
  if (!params) return url;
  const search = new URLSearchParams(params);
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}${search.toString()}`;
}

async function connectMcpServers(
  servers: Record<string, McpServerConfig>,
): Promise<ConnectedServer[]> {
  const connected: ConnectedServer[] = [];

  for (const [name, server] of Object.entries(servers)) {
    if (server.requires) {
      const missing = server.requires.filter(
        (v) => !process.env[v],
      );
      if (missing.length > 0) {
        console.log(
          `[agent] Skipping ${name}: missing env vars: ` +
            missing.join(", "),
        );
        continue;
      }
    }

    const expanded = expandAllEnvVars(server.transport);

    console.log(`[agent] Connecting to ${name} MCP server...`);
    let client;
    if (expanded.type === "stdio") {
      const stdioTransport = new Experimental_StdioMCPTransport(
        {
          command: expanded.command,
          args: expanded.args ?? [],
          env: { ...process.env, ...expanded.env } as Record<
            string,
            string
          >,
        },
      );
      client = await createMCPClient({
        transport: stdioTransport,
      });
    } else {
      const url = buildTransportUrl(
        expanded.url,
        expanded.params,
      );
      const { headers } = expanded;
      const transport =
        expanded.type === "sse"
          ? {
              type: "sse" as const,
              url,
              ...(headers ? { headers } : {}),
            }
          : {
              type: "http" as const,
              url,
              ...(headers ? { headers } : {}),
            };
      client = await createMCPClient({ transport });
    }
    const tools = await client.tools();
    connected.push({ name, client, tools });
  }

  return connected;
}

async function runConfigMode(
  config: AgentConfig,
): Promise<void> {
  const servers = await connectMcpServers(
    config.mcp_servers ?? {},
  );
  const rawTools: ToolSet = {};
  for (const server of servers) {
    Object.assign(rawTools, server.tools);
  }
  const allTools = sanitizeTools(rawTools);
  console.log(
    `[agent] Loaded ${String(Object.keys(allTools).length)} ` +
      `tools from ${String(servers.length)} server(s)`,
  );

  try {
    console.log("[agent] Running config strategy...");
    await runConfigStrategy(allTools, config);
    console.log("[agent] Strategy complete");
  } finally {
    for (const server of servers) {
      await server.client.close();
    }
  }
}

function buildWorkflowRunUrl(): string | undefined {
  const server = process.env["GITHUB_SERVER_URL"];
  const repo = process.env["GITHUB_REPOSITORY"];
  const runId = process.env["GITHUB_RUN_ID"];
  if (server && repo && runId) {
    return `${server}/${repo}/actions/runs/${runId}`;
  }
  return undefined;
}

async function runCodeMode(): Promise<void> {
  const agentToken = requireEnv("AGENT_TOKEN");
  const apiUrl = requireEnv("ARENA_API_URL").replace(/\/+$/, "");
  const taostatsKey = process.env["TAOSTATS_API_KEY"] ?? "";

  const repoUrl =
    process.env["GITHUB_SERVER_URL"] &&
    process.env["GITHUB_REPOSITORY"]
      ? `${process.env["GITHUB_SERVER_URL"]}` +
        `/${process.env["GITHUB_REPOSITORY"]}`
      : "";
  const commitSha = process.env["GITHUB_SHA"] ?? "local";
  const workflowRunUrl = buildWorkflowRunUrl();

  const arenaParams = new URLSearchParams({
    token: agentToken,
    repo_url: repoUrl,
    commit_sha: commitSha,
    workflow_run_url: workflowRunUrl ?? "",
  });
  const arenaSseUrl =
    `${apiUrl}/mcp/sse?${arenaParams.toString()}`;

  console.log("[agent] Connecting to Arena MCP server...");
  const arenaClient = await createMCPClient({
    transport: { type: "sse", url: arenaSseUrl },
  });

  let taostatsClient: Awaited<
    ReturnType<typeof createMCPClient>
  > | null = null;
  if (taostatsKey) {
    console.log(
      "[agent] Connecting to Taostats MCP server...",
    );
    taostatsClient = await createMCPClient({
      transport: {
        type: "http",
        url: "https://mcp.taostats.io?tools=data",
        headers: { Authorization: taostatsKey },
      },
    });
  } else {
    console.log(
      "[agent] No TAOSTATS_API_KEY set, skipping Taostats",
    );
  }

  try {
    const arenaTools = await arenaClient.tools();
    const taostatsTools = taostatsClient
      ? await taostatsClient.tools()
      : {};
    const allTools = sanitizeTools({
      ...arenaTools,
      ...taostatsTools,
    });
    console.log(
      `[agent] Loaded ` +
        `${String(Object.keys(allTools).length)} tools`,
    );

    console.log("[agent] Running strategy...");
    await runStrategy(allTools);
    console.log("[agent] Strategy complete");
  } finally {
    await arenaClient.close();
    if (taostatsClient) await taostatsClient.close();
  }
}

async function main(): Promise<void> {
  console.log("[agent] Starting agent");

  if (existsSync("agent.config.json")) {
    const config: AgentConfig = JSON.parse(
      readFileSync("agent.config.json", "utf-8"),
    );
    await runConfigMode(config);
  } else {
    await runCodeMode();
  }
}

main().catch((err: unknown) => {
  console.error("[agent] Fatal error:", err);
  process.exitCode = 1;
});
