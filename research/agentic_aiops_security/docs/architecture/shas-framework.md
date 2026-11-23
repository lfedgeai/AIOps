# **Scope & History-Aware Agentic Security (SHAS): A Distributed Security Framework**

## **Abstract**

Scope & History-Aware Agentic Security (SHAS) is a distributed architecture that acts as a wrapper around the existing SPIFFE/SPIRE identity framework. While SPIRE effectively answers which workload is calling and which node it runs on, it does not account for shifts in context, leaving agentic systems vulnerable to prompt-injection threats. SHAS addresses these vulnerabilities by embedding an agent’s scope, historical reputation, and intended credentials directly into the security architecture. By enforcing strict profile validations, deterministic policy blocks, and post-execution memory sanitization, SHAS mitigates four primary classes of context attacks: direct prompt injection, indirect prompt injection across trust boundaries, context hacking of shared memory, and temporal attacks that reuse memory or authority after expiration.

![Consolidated flow. The request passes the guardrails, is planned from discovery memory, and revised by Asago. The dispatcher cuts the profile at a trust boundary, Cedar permits publication, and the check value, which covers the long-lived capability token, rides the Bellman countdown into the SPIRE token gate. The gateway refuses a tool sequence that is not a path in the published control-flow graph, and it verifies a transaction token that names the principal. Progent blocks a forbidden call. A cold start or an anomalous call stays in the sandbox until reconcile. The Recorder admits the next brief, quarantines a dangerous chain, wipes the agent, and keeps a same-fault bonded report for forensics.](../figures/shas-flow.svg)

## **1\. The Distributed Threat Landscape**

Agentic workloads fundamentally shift the security paradigm because they are non-deterministic and often required to traverse multiple security boundaries to complete a task. A routine request from a senior leader for a supply chain report may force an internal agent to gather information from highly secure internal sources, followed by queries to remote vendor sites in foreign networks.

This traversal spans several trust boundaries:

| Trust Boundaries | Trust Level | Risk Level | Governance | Traceability/Observability |
| :---- | :---- | :---- | :---- | :---- |
| Core (local corporate LAN) | Full  | Negligible  | Complete  | Total  |
| Intranet (multi-site security boundary) | High  | Low  | High  | High  |
| Federated network (Vendor networks) | Moderate  | Moderate  | Limited  | Low  |
| Internet | None  | High  | None  | None  |

![Four trust boundaries. Trust, governance, and traceability fall from the core, through the intranet and a federated vendor network, to the internet. A fetch at the internet boundary and a write in the core are different profiles.](../figures/shas-trust-boundaries.svg)

When an agent accesses external sites to fulfill a query, it is exposed to the local threat landscape. Without proper containment, an agent could ingest a prompt injection, bring it back across boundaries into the Core network, and poison the swarm’s shared context or memory graph. SHAS treats such boundary crossings as dual-exposure vectors and limits the blast radius of compromised agents by ensuring that even poisoned agents can only execute their strictly assigned, historically validated tasks.

## **2\. Core Architecture & Minting**

To prevent poisoned memory propagation, agents in the SHAS framework are strictly minted for their tasks, and the multi-turn swarm memory is maintained as a highly monitored series of briefs rather than an open communication channel.

![Minting and the clock. The user prompt passes input guardrails. Discovery reports stay in their own memory. Asago revises the plan, output guardrails screen it, the dispatcher cuts the profile along a trust boundary, and Cedar permits publication. The check value binds the system prompt and the long-lived capability token. The per-transaction token that names the principal stays out of that hash. Bellman carries the countdown, and SPIRE issues a short-lived workload token only inside that countdown. Zanzibar is the helper-reuse relation.](../figures/shas-minting.svg)

### **2.1 Orchestration, Planning, and Discovery**

* **The Discovery Protocol:** Before planning, the Dispatcher creates short-lived, hardcoded, read-only agents (e.g., using Ansible) to poll known network resources for available data based on the current role.

* **Separated Discovery Memory:** The resulting Completed Work Order Reports (CWORs) are stored in a distinct memory space completely separate from workload memory, providing the Orchestrator and Dispatcher with untainted topological context.

* **Policy-as-Code Revision:** The Orchestrator develops a granular plan and runs it against a policy-as-code engine (such as Asago).

* **Iterative Auditing:** The plan is iteratively revised until it passes the policy audit or reaches an administrator-defined turn limit.

* **Input and Output Guardrails:** The initial user prompt passes input guardrails before agent assignment, and the Orchestrator's plan passes output guardrails before the Dispatcher assigns it, ensuring that a request attempting to leave the assigned duty fails immediately.

### **2.2 Minting and Policy Publication**

* **The Grain Rule & Security Boundaries:** The Dispatcher prefers utilizing multiple existing single-job agents over minting a single agent capable of spanning multiple trust boundaries.

* **External Policy Evaluation:** Publication of a newly minted profile is governed by an external authorization engine (such as Cedar).

* **Boundary Enforcement:** A profile whose actions attempt to span two distinct trust boundaries (e.g., core writes and internet fetches) receives no permit and is never published.

* **Capability Token:** Each minted agent receives one capability token. It names the rights in that profile, including the tools the agent may call and the control-flow graph it may follow. The token is long-lived for the commission and is not reissued on each request. A revocation, or a countdown that expires, withdraws it with the agent. A core write and an internet fetch are different tokens because they are different profiles.

* **Immutable System Prompts:** Once minted, the agent's system prompt, creation metadata, and capability token generate a public check value.

* **Check Value Binding:** The check value binds those three together. A direct injection that replaces the formulation fails the check. A presented capability that is not the published token fails it as well. The transaction token that names the principal is minted per transaction and stays out of this hash.

* **Best-Guess Addendum:** A "best-guess" workflow is attached strictly to the first user prompt of a newly minted agent, purposefully excluded from the system prompt and from the capability token, so it cannot alter the published check value.

### **2.3 The Encrypted Bellman Protocol & Temporal Security**

* **Cryptographic Broadcasts:** The Bellman Protocol is a cryptographically signed broadcast that informs edge nodes of valid, commissioning, and decommissioning agents.

* **Volatile Revocations:** Because revocations are volatile, they trigger an immediate Bellman publication rather than waiting for a scheduled poll.

* **Decommissioning Countdowns:** The broadcast carries a decommission countdown for unused agents, which can be reset by subsequent lists.

* **Token Issuance Gates:** SPIRE will only issue its short-lived workload token if the remaining countdown outlasts that token's lifetime and a resetting publication occurs before the countdown expires. The gate does not reissue the capability token. It does not mint the transaction token. That token is issued per transaction by the trust domain's transaction token service.

## **3\. The Execution Pipeline & Zero Trust Containment**

SHAS enforces zero trust principles by placing enforcement points directly on the request path, ensuring anomalous or unauthorized behavior never reaches the primary database.

![Execution, containment, and memory. The gateway asks five questions, hashes the system prompt together with the capability token, and verifies a transaction token that names the principal while the agent's own identity remains the actor. It refuses a tool sequence that is not a path in the published control-flow graph. Progent blocks a forbidden tool call. Cold starts and anomalous permitted calls stay in the sandbox until a dry run and an in-toto layout allow reconcile. The Recorder admits a safe report as the next brief, quarantines a dangerous chain, wipes the agent, and keeps a same-fault bonded report for forensics.](../figures/shas-execution.svg)

### **3.1 The Attestation Gateway**

When an agent reaches a resource gateway, the system hashes the presented system prompt together with the presented capability token and compares the result with the check value in the local Bellman registry. The gateway asks five gating questions. SPIFFE/SPIRE answers which workload is calling and which node it runs on. The Bellman countdown answers when the work may be done. The capability token and the published profile answer what the agent may do. A transaction token answers who the work is for.

1. Is this one of the organization’s approved agents?

2. Where did this agent come from?

3. When is the work supposed to be done?

4. What is this agent allowed to do and request from this resource based on its profile?

5. Who is this agent working on behalf of?

Question 5 is a short-lived transaction token, issued for one transaction inside one trust domain [@rfc8693; @ietf-txn-tokens]. The agent authenticates to that trust domain's transaction token service with its own SPIRE workload identity. The service decides that this agent may act for this principal and mints the token. Delegation keeps both identities: the subject is the principal, and the actor remains the agent [@rfc8693]. The transaction-token profile carries the same split, with `sub` for the principal and `req_wl` for the workload that requested the token. `aud` names the trust domain, and a gateway outside that domain refuses the token. `txn` binds the token to one transaction. `scope` is the purpose the service assigns, and it may be narrower than the grant the agent presented. The token lives for minutes. It is a separate credential from the long-lived capability token, and the check value does not cover it. One logical transaction token service exists per trust domain. The gateway verifies the presented token. It does not choose the principal. A core write and an internet fetch are different trust-domain audiences, so a token minted for one does not authorize the other. The WIMSE architecture treats this credential as authorization context bound to the workload identity [@ietf-wimse-arch]. RFC 8693 is the published standard for the exchange. The transaction-token profile and the WIMSE architecture text are Internet-Drafts, and agent-specific delegation-chain drafts are not the baseline this gateway uses. SPIRE's Delegated Identity API is a different mechanism: it returns another workload's identity. Question 5 keeps the calling agent's own identity as the actor.

The profile published on the Bellman list carries a control-flow graph, built in the sandbox from benign tool traces [@agentguardian2026]. A node is one occurrence of a tool. An edge exists only when one tool directly follows another in a benign sequence. At the gateway, a tool call proceeds only when the sequence of previous tools is a path in that published graph. A missing edge is a refusal of what the agent may do at this resource. The call does not run, and it is not an anomaly sent to the sandbox. The gateway does not add edges from the live workload, from workload memory, or from a quarantined report. The capability token names that graph, and the check value binds the token together with the system prompt.

### **3.2 Reputation, Monitoring, and the Sandbox**

* **Agent Reputation Database (ARDB):** The ARDB tracks recent behavior of agents on local nodes (retained from hours to weeks) to establish a baseline of expected work.

* **Fingerprinting Cold Starts:** Every cold-start agent must run its first task in the Task Sandbox to establish its initial fingerprint and base run for the ARDB.

* **Policy Enforcement Points (Progent):** Tool calls are deterministically evaluated against published policy before execution; calls forbidden by policy (e.g., bulk deletions outside the permitted scope) are blocked entirely.

* **Sandboxing Anomalies:** Permitted calls that are anomalous (outside the established baseline) are transferred to the Task Sandbox by the Agent Monitoring Client.

* **CRDT Shards and Dry Runs:** Inside the sandbox, writes and deletes are recorded as local CRDT shards. A dry run verifies these shards to ensure the effects are not destructive or breaking.

* **Effect Reconciliation (in-toto Layouts):** Shards are only reconciled into the primary database once the agent's profile is validated and the dry run proves the effects match the allowed layout. Untrusted data retrieved by the agent (e.g., via CaMeL control flow protections) cannot choose or force these effects.

### **3.3 Helper Reuse and Subcontracting**

* **Subset Relations (Zanzibar):** Helpers carry stricter limits than primary agents and are requested from the Dispatcher.

* **Equal or Reduced Scope:** A later task may only reuse a remembered helper if the scope of the new task is equal to or reduced compared to the helper's minted profile.

* **Behalf of the same principal:** A helper authenticates with its own workload identity. The transaction token service in that trust domain may issue a token whose subject is the same principal and whose actor is the helper. The scope is no wider than the helper's capability token. The helper presents that token, rather than the caller's.

## **4\. Post-Execution, Memory, and Forensics**

The architecture strictly regulates how a completed task becomes context for future tasks to prevent poisoned memory propagation.

### **4.1 The Recorder and Drift Validation**

* **Report Generation:** After execution, the agent returns a structured JSON Completed Work Order Report (CWOR) detailing retrieved information, tools called, queries made, and resources accessed.

* **Drift and Adherence:** The Recorder compares the report against organizational policy and the system prompt. The drift result checks for adherence; a factual error inside the assigned duty passes, but exfiltrating data outside the system prompt fails adherence and is withheld.

* **Sanitization:** The Recorder sanitizes restricted information and runs the CWOR through guardrails aimed at direct injections.

### **4.2 Semantic Auditing and Quarantine**

* **Session Audits (JEV Models):** CWORs in a multi-turn workflow carry session tags, allowing a classifier (like a JEV model) to semantically audit the entire chain before adding it sequentially to memory.

* **Quarantine:** A session chain read as dangerous is instantly stopped, isolated, quarantined, and flagged for human review. Quarantined reports are never used as the next brief.

* **Context Wiping:** Once a report is submitted, the specific agent's context, tool calling, tool cache, memory, and indices are completely wiped before returning to the warm pool.

### **4.3 Redos and Bonded Agents**

* **Workload Redos:** If a CWOR is rejected, the Recorder prompts the Dispatcher to reattempt the task with a different, tighter minted agent.

* **Bonded Agents:** To work around failed CWORs, the Dispatcher utilizes one-shot Bonded Agents.

* **Synchronous Hashing:** The entire bonded prompt is hashed and published synchronously via the Bellman Protocol exclusively to the nodes involved in the previous failure. That hash also includes the bonded agent's capability token, which is one-shot and read-only. The bonded agent's transaction token is minted for that read, for the same principal, and stays out of the hash.

* **Read-Only Forensics:** Bonded agents are strictly read-only and disappear immediately after their sub-task is completed.

* **Avoidance Routing:** If a bonded agent's report reveals the same fault, it is retained for forensics but never added to memory. The Orchestrator is instructed to avoid the source, and if the workflow cannot proceed without it, the task fails cleanly.

## **Conclusion**

Scope & History-Aware Agentic Security ensures that no single agent possesses the ability to execute an entire cybersecurity kill chain. By building on top of SPIFFE/SPIRE—which effectively authenticates *who* the agent is and *where* it operates—SHAS systematically authenticates *how* the agent is formulated, *what* its exact profile permits, *which* context it is allowed to retain, *when* it is authorized to act, and *for whom* it acts. The last of these is a short-lived transaction token. Its subject is the principal, its actor remains the agent's own workload identity, and a gateway outside that token's trust domain refuses it. Through cryptographic prompt integrity, zero-trust sandbox containment, and aggressive memory sanitization, SHAS transforms distributed agentic workloads from systemic risks into reliable, scope-bound components.  
