# AIOps - for distributed environments and edge

The aim of this AIOps project under the LFEdge umbrella is to help deliver AI supported operations for heterogeneous and complex distributed environments such as edge.

#Team Meetings
## Timings
AIOps Sync Call - weekly
Every Tuesday, · 13:30 – 14:10 - Time zone: Australia/Sydney (Summer)
Every Tuesday, · 9am – 9:40 - Time zone: India
Every Monday, · 4:30pm – 5:10pm - US/WestCoast/LA (Winter)
Every Monday, · 9:30pm – 10:10pm - US/EastCoast/NY (Winter)

##Google Meet joining info
Video call link: https://meet.google.com/hps-nrkm-esn
Or dial: ‪(AU) +61 3 8594 7760‬ PIN: ‪478 062 271‬#
More phone numbers: https://tel.meet/hps-nrkm-esn?pin=1051369562855
Or join via SIP: sip:1051369562855@gmeet.redhat.com


There are three workstreams on the highest level:

## Research
The motto inside this folder is "Anything AIOps Goes". Here's the home for your wildest ideas to try out, code, test and report back on. Ideally everything built on open source technology or at least free to use and openly available software or models. Examples are  evaluating different algorithms or models for anomaly detection, trying different agentic architectures like agents for logs, metrics, traces VS agents for each system component or layer of the technology stack.

## Test Harness
The test harness is where we assemble different components of AIOps solutions to test and benchmark against each other. This can be sub-component test harnesses like for anomaly-detection or a full end-to-end AIOps solution cycle from data/signal collection, signal correlation, anomaly detection, root cause analysis, and remediation.

## Demo
The demo is where the contributors assemble working demos and products to showcase.

## AIOps workflow and high level component overview
![Alt text](AIOpsOverview.png)

The 'Recommend' arrow has several human in the loop (HIL) remediation maturity levels:
- Level 1: Assisted Questioning (lowest level of maturity)
- Level 2: Migitgation Solution Generation (checklist of tasks)
- Level 3: Command Recommendation (low level command recommendation according to the checklist)
- Level 4: Script Generation (entire script of all commands)

before we move into the Autonomous Operations realm with
- Level 5: Automatic Execution (no HIL)

(source https://arxiv.org/pdf/2507.12472)

## Why it matters
- **Manage Complexity** – Distributed and heterogenous environments can be complex and hard to manage. Hece we are utilizing AI to help secure, fault-find and manage those environments.
- **Adopt** – Due to the vast differences distributed environments have wrt connectivity, workloads and general resource constraints, AIOps trains and finetunes the AI models 'on the job', learning and adopting to changing environments
- **Scale** – Only AIOps can enable growing distributed environments to remain manageable and secure.

Other factors are cost reduction, improving system perfomance, and decreasing incident resolution time.

## AIOps Component Overview
### Bootstrapping
- Infra & Runtime (Kind | MicroShift | Fedora)
- Setup of secure communication infrastructure
- Deployment, config & localisation
### UI/API
- Inference | Model selection | Model training | Testing | Rollout
### MLOps
- Workflows
### AI Engineering
- AI Model development | training | testing | validation | comparison
### Data Platform & Storage
- Data engineering/science | data Pipelines/workflows | data cleansing | Analytics | data collection | storage (Iceberg|MinIO)| data entity models | data query 
###  Security & Trust (cross-cutting) (Lead: Ramki Krishnan)
- Platform | Model | MLOps
- Encrypted data transfer | storage
- Workload identity (SPIFFE | SPIRE | CitadelAI)
- Geofencing | non-repudiation | Verifiability
### Inference
- Model inference engine
- Authentication | authorization
- Predictive Models | LLM | RAG (system documentation, data entity information) 
- Alerting
### Remediation
- Slack | Chatbot | Agents | Automation
### Telemetry
- Signal collection | OTEL
- Model performance (accuracy), energy consumption,optimized inference
- Logging | tracing | metrics 
### Other
- Documentation | web-site




## Team
- [Andreas Spanner](https://lf-edge.atlassian.net/wiki/people/59fc56048499730e3412487a?ref=confluence) (Red Hat) (Lead) 
- [Ramki Krishnan](https://lf-edge.atlassian.net/wiki/people/557058:c8c42130-9c8b-41ae-b9e2-058af2eff879?ref=confluence) (Intel) Component Lead - Security
- [Jey Paulraj]  (Red Hat) Component Lead - Demo 
- [Clyde Tedrick] (Zaga)
- [Giri] (Zaga)
- [Prathiba] (Zaga) Component Lead - Data
- [Stas Kozlov] (Red Hat) 
