# ContextManager

The ContextManager is a class that manages the context of a model that's working
in a given repository. This is inspired by the ring buffer used by
[llama.vim](https://github.com/ggml-org/llama.vim) along with the [back-end
support provided by llama.cpp](https://github.com/ggml-org/llama.cpp/pull/9866):
`--cache-reuse` and `"cache_prompt": true`

We assume the user has a GPU always at the ready within their own laptop or
desktop, that the ContextManager has exclusive use of and control over. We
assume that the dominant task that will be undertaken is editing *code*.

The basic idea here is to treat the input KV cache as a buffer that can be
selectively edited more like the [way `malloc()`
works](https://en.cppreference.com/w/c/memory/malloc). This way while editing
the file, the ContextManager will maintain a set of relevant information to the
task at hand by allocating, de-allocating, moving, and expiring information in
the context. When the context needs to change, we can just overwrite a section
of the context with null (or spaces) and keep a "free space list" of the context
so that when something is allocated, we can choose a location to put it in as
quickly as possible, without having to re-tokenize and re-process the entire
prompt.

Context will be delimited by XML tags (assuming the code being edited is not
itself XML).

## Context Types

Context will be represented by an XML-style tag with attributes that describe
the type of scope. For instance
```
<context type="filesystem", name="root", editable=true>
foo/
foo/__init__.py
main.py
requirements.txt
</context>
```
or
```
<context type="package" name="re", editable=false>
re.A re.M re.Scanner() re.error() re.search() re.ASCII re.MULTILINE re.U
re.escape() re.split() re.DEBUG re.Match() re.UNICODE re.findall() re.sub()
re.DOTALL re.NOFLAG re.VERBOSE re.finditer() re.subn() re.I re.Pattern()
re.X re.fullmatch() re.IGNORECASE re.PatternError() re.compile() re.functools
re.L re.RegexFlag() re.copyreg re.match() re.LOCALE re.S re.enum re.purge()
</context>
```
Types of context may include:

1. `filesystem`: filesytem structure
2. `class`: class interface definition
3. `scope`: names within the current scope
4. `package`: names within the `package` scope such as `re.match`.
5. `documentation`: non-code documentation
6. `file`: an entire file (e.g. json test case?)
7. etc.

As the user moves between files or adds code, the ContextManager is responsible
for adding and removing relevant context. For instance if the user adds an
`import re` (regular expression library) statement in his code, the above
context might be added. This example is just the output of `dir(re)` and could
be made a lot better, e.g. by adding type hints and function signatures.

Because the user will likely be editing the same file for some time, when we
pre-load context, we leave it there. Thus we absorb the tokenization time of the
context only once, and thereafter re-use it until it is de-allocated.

The ContextManager can also respond to other agents to modify context. These
other agents would be asynchronously seeking out additional context such as
pulling documentation from the web, RAG retrieval data, dependencies, AST, etc,
and asynchronously adding it to the context between processing rounds.

## Editing Context

In addition to the above context, there will be a `<editing>` block that
includes the entire contents of the smallest reasonable scope piece of code the
user is currently working on. What this is depends on the user's current scope
and where the cursor is located.  For example:

1. (entire) Function being edited
2. Class definition with function signatures and class member variables but no
   implementations
3. File-level list of variables, types, function definitions, and any code
   outside a function (e.g. python __main__)

For instance:
```
<editing type="function" file="foo/__init__.py", name="baz">
def baz(s: str):
    <|FIM_MIDDLE|>
    return bool(re.match('bar', s)
</editing>
```

Assuming the user has `import re` at the top of `foo/__init__.py`, the context
example in the previous section giving the globals and functions in the `re`
package would be present. If the `import re` line were deleted, this piece of
context would be removed. `re` is a less interesting example because it has a
stable interface that every LLM has been trained on, but imagine that was an
obscure package that the LLM was not trained on.

## Instructions

For large enough LLM's, we will also write prompt that describes the above
structure and what the LLM is intended to do with it. This is very similar to
FIM and I think fine tuning these instructions and adherence to them could be
very valuable, especially for smaller LLMs.

For example:
```
<instructions>
Above is a series of <context> tags describing the code a user is working on in
the {language} programming language.  These have a "type" attribute that may be
one of:
1. `filesystem`: filesytem structure
2. `class`: class interface definition
3. `scope`: names within the current scope
4. `package`: names within the `package` scope such as `re.match`.
5. `documentation`: non-code documentation
6. `file`: an entire file
Additionally each context block has a "name" attribute that you must use when
referring to it, and an "editable" flag indicating whether you are allowed to
modify it.

There is also a <editing> block that the user is currently working on. This has
just been updated with an edit the user has made. The location of the user's
cursor is given by the tag <|FIM_MIDDLE|>.

You are to examine the context for consistency, relevance, clarity, good code
style, and syntactic correctness. You can make suggestions for the
<editable> block or any context block with the "editable" flag set to true.

Output a list of blocks enclosed by the <suggestion> tag. The attributes on the
<suggestion> tag must include the name of the context block it came from or
`name="editing"` if you're suggesting a change to the <editing> block. You can
output multiple suggestions to ensure self-consistency.

Finally output a block <summary> that contains a description of the change
currently being made. Do not output anything other than the XML blocks described
above.
</instructions>
```

## Variants and Considerations

Aider seems to indicate that many LLM's have difficulty outputting diffs. It
uses "whole" diff format, where the LLM reproduces the entire file, a "diff"
format which consists of SEARCH and REPLACE blocks, and a "udiff" format which
is the unified diff format with +/- to add/remove lines. It seems LLMs are bad
at numbering lines and will generally get the line number markers in the unified
diff format incorrect. As we're strictly controlling the context and not giving
the LLM the entire file, for it to output an entire function is faster than
having it output an entire file. However for larger inputs we might want to
consider something more diff-like. (I strongly suspect that half the problem
here is in Aider itself and its focus on prompt engineering instead of actual
engineering, but I digress...)

### ln-diff format

A [suggestion made to me by an
LLM](https://chatgpt.com/share/6842a722-a6dc-800f-ade6-8ccb5b489bc0) was to
number lines, sometimes called "ln-diff". Consider:
```
<editing type="function" file="foo/__init__.py" name="baz">
L1 def baz(s: str):
L2    <|FIM_MIDDLE|>
L3    return bool(re.match('bar', s)
</editing>
```
an instruction can be given:
```
In the suggestion block only output the lines that you want to change. Each line
must be preceded by its line number.
```
which might result in a suggestion:
```
<suggestion type="function" file="foo/__init__.py" name="baz">
L2    logger.log(f"baz({s})")
</suggestion>
```
This minimizes the output length and will result in faster responses. However we
will have to do some testing to see if common LLMs are good at following
instructions.

### AST diff format

A [suggestion made to me by an
LLM](https://chatgpt.com/share/6842a722-a6dc-800f-ade6-8ccb5b489bc0) was to
use AST transformations, such as those used by
[semanticdiff](https://semanticdiff.com/blog/language-aware-diff-how-far/)
