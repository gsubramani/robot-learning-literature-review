# Task: Review arxiv 2205.13147 (Matryoshka Representation Learning)

## Goal
Review the paper and explain how the model compression works, specifically how to apply it to LSTM and Transformer layers.

## Status
Complete.

## Key Finding
MRL is a **representation compression** technique, not a weight compression technique. It trains a single embedding where the first m dimensions are independently useful, enabling adaptive deployment at varying granularities.
