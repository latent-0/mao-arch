"""Real external benchmarks for the joint-embedding handoff gate.

The core pipeline trains only on synthetic workflow templates (mao/datagen.py).
This package instantiates task graphs from a *real* external benchmark so the
trained gate can be evaluated zero-shot on task topologies and vocabulary it
never saw in training.
"""
