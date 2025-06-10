# BRaid AI eDitor

The BRaid AI eDitor will be a an agentic command line editor similar in spirit to [Aider](https://aider.chat/) and [Claude Code](https://www.anthropic.com/claude-code) but taking an agentic approach, and focusing on local LLM's that are run on the user's GPU, while maximizing its speed and context use.

# Agentic Components

There will be a series of agents which work in concert to answer the user's query.

* [Orchestrator](https://github.com/braidpool/agent_orchestrator) -- watches agents for crashes, failures, repetition, restarts, etc.
* Architect (large LLM) -- develops the initial plan and tracks its progress using a TODO list
    - add TODO
    - complete TODO
    - re-evaluate plan with new information
* Editor (medium-small LLM) -- actually writes code
    - may have multiple of these depending on the complexity of the task
* Evaluator (medium-small LLM) -- evaluates whether the code written satisfies the tests
    - applies diffs
    - lints results
* [ContextManager](https://github.com/braidpool/Bitcoin-AI/blob/main/ContextManager.md)
    - manages context chunks
    - content based addressing (deduplication)
    - save/restore (e.g. queries by Architect, Editor, Evaluator will often re-use the same chunks of code)
* DocumentationFinder -- finds documentation using RAG or web searching
    - includes ChatBTC RAG
    - docstring/inline documentation extraction
* RepositoryTracker -- keeps track of code including filesystem layout, git integration, and AST/GraphRAG
    - "What does this change affect?"
    - "Which file needs to be edited?"
    - "Who calls this function?"
    - "What's in scope to this function?"
    - "What's the function signature of this thing I need to call and what does it do?"

# Simplifications

* Use a standardized format for feeding code to avoid variance in the output. The code should be stripped of comments (they can be in a separate documentation context chunk) and formatted with a linter.
* Use AST parsing to apply diffs, so as to be insensitive to "diff format".
* Context Chunks, not prompts: select the context chunks you need and add instructions.

# Evaluation

* We need a way to find small models that work well
  - Start with simple tests (change line 3, change variable x to y etc)
  - Refine sampling parameters to remove repetition
  - Find where they fail
* Evaluate standard coding benchmarks (exercism)
* Find large-codebase evals
* Find documentation/hallucination benchmarks

# Questions

* Using SGLang/vLLM, can KV cache chunks be moved between models? e.g. if we're running Qwen3:8b for orchestrating and Qwen3:0.6b for applying diffs, can they share context chunks?
    - I think so because I think speculative decoding uses it
