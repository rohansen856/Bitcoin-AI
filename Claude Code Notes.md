# Claude Code Notes

Claud Code is an orchestration layer on top of the Anthropic Opus and Sonnet models. It adds a number of tools that the AI can call. You can find these in `~/.claude/projects/*/*.jsonl`. These are:
```
find ~/.claude/projects -name \*.jsonl -exec jq -s '.[] | select(.message.role == "assistant")
                                                        | .message.content[]?
                                                        | select(.type == "tool_use")
                                                        | .name' {} \; \
    | sort | uniq | sed 's/"//g'
```
1. Bash
2. Edit
3. Glob
4. Grep
5. LS
6. MultiEdit
7. Read
8. Task
9. TodoRead
10. TodoWrite
11. WebFetch
12. WebSearch
13. Write

It uses a "planning" mode that creates TODO files using `TodoWrite` (these are in `~/.claude/todos`) and progressively checks them off.  Here's an example:
```json
[
  {
    "content": "Extract diff formats from original tuner.py",
    "status": "completed",
    "priority": "high",
    "id": "1"
  },
  {
    "content": "Add diff format data structures to test_loader.py",
    "status": "completed",
    "priority": "high",
    "id": "2"
  }
]
```

# Claude Successes

## Context Compaction

In the lower right Claude has a message `Context left until auto-compact: 37%`. When its context window is reached, it "summarizes" its context and clears it. I like the tracking of context usage (which Aider does not have and silently fails). I'm not sure if it keeps TODO's across compaction, but it must.

### ⇒ Implication

Context management is important, but Claude's method is flawed. One of the projects I was having it implement was the [ContextManager](https://github.com/braidpool/Bitcoin-AI/blob/main/ContextManager.md) idea for [llama.cpp](https://github.com/ggml-org/llama.cpp). We will need to do compaction sometimes, but an even better idea is to manage context so that only relevant things are put in the context. For instance, pass it only the function definition that needs to be modified, not an entire file. Using Paged Attention, we can swap in and out pieces of context relevant to the question, and swap context to system RAM as well, maximizing our use of the context window.

## Truly excellent high-level planning

Like most AI tools, Claude is great if it can one-shot your problem, and breaks down the further you use it. Opus is a truly impressive model.

### ⇒ Implication

Consider using larger models like Opus for initial planning, and smaller models to execute the plan.

## Explicit review finds failures

If you ask Claude to review the changes and ask if they implemented the initial instructions, it is pretty good at figuring out whether it does or not. But you have to ask explicitly. I suspect this is because they want to bill you for the extra tokens in the review process.

### ⇒ Implication

Self-review needs to be a default part of the pipeline. The LLM shouldn't report success unless it has been reviewed.

## Command line is simple

Using Claude from the command line is simple and effective. You don't have to figure out VS Code plugins, Github integrations, etc.

### ⇒ Implication

Do one thing and do it well. Don't try to be everything to everyone. Too many AI projects are failing by trying to do too much.

# Claude Failures
## Claude uses grep and find instead of an AST or RAG

You notice in the tool list that it has Bash, Glob, Grep, LS, and it uses these a **lot** to figure out what to edit. It doesn't know what's in any file and often can't find it. It uses `find... -exec grep ... \;` a lot as well.

### ⇒ Implication

Keeping the filesystem layout and an AST based class/module tree would be incredibly valuable. What Claude is trying to do most of the time is find "where is this function" or "who calls this function".

## Claude fails catastrophically on large codebases

Claude generally doesn't have any sense of the larger organization of a codebase, and can't create or find it with Bash, Glob, Grep, and LS. It can find files that mention a particular function, and it uses grep to figure out which files are calling this function. Claude does MUCH better if it can fit the entire codebase in its context window, but its context management is poor.

### ⇒ Implication

The AI needs to keep a high-level view of the codebase, including the filesystem layout of relevant code, and an AST or GraphRAG understanding of what is where. If we have to use grep we've failed somewhere before that.

## Context Compaction can lose old context

One of the reasons I think Claude starts using grep and find a lot is that it has done several rounds of compaction, and early data that it started with, like the filesystem layout, gets lost in the compaction.

### ⇒ Implication

When context is compacted (which it must be), high-level orchestration peices should be spared from compaction, such as the original architecture plan, filesystem layout, TODOs in progress, etc. "Summarizing" these will lead to brain damage.

## Claude still hallucinates function signatures

Just like most AI tools, Claude easily hallucinates the arguments to functions. But worse, it hallucinates this **within** your own repository, because it uses grep to figure things out.

### ⇒ Implication

The fundamental thing Claude is trying to figure out is "which files/functions are involved in this change" and "who calls this function that will also need to be modified". It's way easier and more precise to use an AST or GraphRAG for this, than getting the AI to parse the output of grep.

## Web searching is largely a waste of time

Web searching generally fetches very large documents that quickly blow out Claude's context, and it has trouble summarizing. When used in a "planning" mode, it's fairly good, but it's unable to effectively retrieve documentation or dependency information using web searches.

In coding mode, Claude is trying to compensate for the changes in dependencies that have occured since the end of its training. It generally fails at doing this unless explicitly asked, and then it still fails to obtain relevant documentation.

### ⇒ Implication

Web searching is an inefficient and ineffective way to find documentation. We should use the dependency management of the language and build system instead, and find the exact thing being looked for using non-AI tools.

## Hallucinated solutions

In one of my experiments it hallucinated an entire solution, and I wasted 2 days debugging why it didn't work. It was extremely enthusiastic about the idea and claimed it could create it. But its code was full of TODO and "in a real solution" and "we need access to internal data structures" in comments. Instead of investigating the code in the **same** repo which had the data structures it needed, it crated an entirely independent module that did not use the parent repository's code. It filled a simulated KV cache with static data, and then copied it around, and silently, wasn't doing anything at all. It reported over and over that it had solved the problem and implemented the outline.

### ⇒ Implication

It didn't know *where* to find the data structures it needed. The initial instructions had this information. We can easily provide the relevant data structures as context and not make the LLM figure it out for itself.

## Tool use is slow

Claude is primarily architected around tool use. This means there are a lot of round trips in asking the LLM to use a tool/use tool/tool response, and it's slow. High level orchestration is lost.

### ⇒ Implication

Don't use tools or use them sparingly. At some level we need to ask "what do you need to solve this problem" and that can use tools. But often we can figure that out through software engineering rather than asking the LLM (such as, "which function needs to be modified?") Then we need to use what is found to ask it to solve the problem. Fundamentally we want to know:
1. What do you need to execute this task/solve this problem? ( --> tools / engineering)
2. Is this enough to execute this task/solve this problem? ( --> tools )
3. Solve the problem ( no tools )
4. Did you solve the problem? ( No? GOTO 1 )

## No systematic testing

Claude often tests its code through tool use, compiling and executing the resulting code, and sometimes even writing very long Bash tool use that actually calls python -c with a long program. It even will write tests to a python file and then ask to execute it. This is somewhat dangerous unless the user is paying attention and reviewing the python code, which becomes impractical.

### ⇒ Implications

We should be linting and syntax checking code edits before they even hit the compiler. Stop bad edits early and make it correct them immediately. There's nothing worse than the LLM parsing 100 pages of compiler error logs because they forgot an `import` or `#include`.

Unit tests should be systematic. This is more about orchestration than the LLM. But I think for nearly every change a unit test should be written to demonstrate that it does what it's supposed to. This should be a different LLM invocation though. We can feed the "test writer" agent the instructions, original and changed code and ask it to write a test that exercises the change. Too many instructions confuse LLMs and asking it to write tests at the same time as the code is likely to fail.
