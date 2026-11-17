# Executive summary: consolidated vs siloed telemetry storage

Tracking issue: [lfedgeai/AIOps#45](https://github.com/lfedgeai/AIOps/issues/45)

## What the issue is about

When software runs in production, it constantly writes three kinds of records about itself: logs (a diary of events), metrics (numbers such as CPU use or error counts), and traces (the path one request takes through all the services). The group is building an AI assistant that reads these records to spot problems, find the cause, and fix it. The issue asks one question: does it matter how these records are stored?

## What we test

There are two ways to store the records:

- One shared store. All three kinds of records go into one big database (ClickHouse). This is like keeping all your paperwork in one filing cabinet.
- Separate stores. Each kind of record has its own system (Thanos for metrics, Loki for logs, Tempo for traces). This is like having three cabinets in three different rooms.

The work has two parts.

First, a speed test without AI. We ask both setups the same questions and time the answers. With three cabinets, a program must match the three answers, and the speed of that step depends on how the program is written. So we time the cabinets and the matching step separately, and we write the matching step in two languages (Python and Go).

Second, a test with AI. We run the same demo shop app and break it on purpose in the same ways. Each AI model works as two assistants:

- One assistant can only use the three separate cabinets. It must ask each cabinet and connect the clues itself.
- The other assistant uses the one shared cabinet.

Both assistants get the same instructions, the same background information, and the same problems. Each assistant must actually fix the problem. We use three AI models: two open models (GLM 5.3 Flash and Qwen3.8 27B) and one leading commercial model (Claude Opus 5.5). We measure four things:

1. How fast the assistant notices the problem.
2. How easily the assistant links the three kinds of records together.
3. Whether the assistant finds the right cause.
4. Whether the assistant fixes the problem, and how fast the system recovers.

## Why

To find a cause, the assistant usually has to connect clues across all three kinds of records. With one cabinet, it can pull everything in one request. With three cabinets, it must ask three times and match the answers itself. That can be slower and easier to get wrong. The group wants hard numbers on whether this difference is real and large enough to matter. Then the group can recommend one storage design for this kind of AI assistant, or show that the choice makes no real difference.
