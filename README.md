# Persistence Gate

A prototype memory-control layer that sits between retrieval and downstream decision-making.

It does not replace search, embeddings, databases, or LLMs. It controls which retrieved information is allowed to influence a decision.

Core loop:

```text
retrieve -> score -> gate -> use -> feedback -> update
```

## Plain English

Normal retrieval systems do this:

```text
Find relevant information -> use it
```

Persistence Gate does this:

```text
Find relevant information -> check if it is safe/useful/current -> allow or block it -> learn from feedback
```

The project is a small Python software prototype for testing the idea that data should not be allowed to influence a system just because it is relevant.

## Current status

Research prototype. Not production software yet.

Supported claim so far: matched synthetic simulations show persistence-aware v2 beats ordinary retrieval on net utility by reducing harmful retrievals and burden while maintaining useful retrieval. The remaining design problem is abstention-aware retrieval: deciding when memory should not be used at all.

## What the demos show

### Sample demo

Compares:

1. **Ordinary top-k retrieval**: use the most relevant memories.
2. **Persistence Gate**: use only memories that pass relevance, usefulness, risk, burden, and validity checks.

### Real GitHub corpus demo

Builds a small corpus from real GitHub repository metadata and README text, then compares ordinary top-k retrieval with Persistence Gate.

It shows:

- what ordinary top-k would have used
- what Persistence Gate allowed
- what Persistence Gate blocked
- which blocked memories ordinary top-k would have used
- a quick interpretation of whether the gate changed influence

## Package layout

```text
src/persistence_memory/
  models.py        # data models and states
  scorer.py        # persistence scoring
  controller.py    # retrieve-score-gate-feedback loop
  store.py         # in-memory store
  evaluation.py    # metrics
examples/
  github_real_data_demo.py   # bundled sample corpus demo
  github_corpus_demo.py      # real GitHub API corpus demo
  sample_corpus.jsonl
tests/
  test_controller.py
  test_github_corpus_demo.py
```

## Run in Google Colab

```python
!git clone https://github.com/JayCamm/persistence-gate.git
%cd persistence-gate
!pip install -e ".[dev]"
!pytest
!python examples/github_real_data_demo.py --sample
!python examples/github_corpus_demo.py
```

If you are re-running in the same Colab session:

```python
!rm -rf /content/persistence-gate
%cd /content
!git clone https://github.com/JayCamm/persistence-gate.git
%cd persistence-gate
!pip install -e ".[dev]"
!pytest
!python examples/github_corpus_demo.py
```

## Run locally

```bash
git clone https://github.com/JayCamm/persistence-gate.git
cd persistence-gate
pip install -e ".[dev]"
pytest
python examples/github_real_data_demo.py --sample
python examples/github_corpus_demo.py
```

## Optional: use a GitHub token

The real GitHub corpus demo uses the public GitHub API. For higher rate limits, set a token first:

```bash
export GITHUB_TOKEN=your_token_here
python examples/github_corpus_demo.py
```

## Design principle

Data should not merely be retrieved because it is relevant. It should be allowed to influence the system only when its expected current value exceeds its risk, burden, and abstention baseline.
