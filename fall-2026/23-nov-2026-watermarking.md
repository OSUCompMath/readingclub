# Text Watermarking: From Red/Green Lists to textGrain

A tutorial for mathematicians on how language models embed detectable statistical signals in generated text. Starting with next-token sampling and keyed randomness, we explain generation and detection for red/green watermarking, GumbelSoft, and textGrain. Worked examples show how textGrain uses optimal transport to preserve token probabilities on average while controlling entropy loss.

[ML Club](https://hanveiga.com/ml-club/)

## References

- Kirchenbauer et al. (2023). [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226).
- Jang, Gu, and Poole (2017). [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144).
- Fu et al. (2024). [GumbelSoft: Diversified Language Model Watermarking via the GumbelMax-trick](https://aclanthology.org/2024.acl-long.315/).
- Li et al. (2026). [textGrain: Entropy-Calibrated Watermarking for Language Model Text](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf).
- OpenAI (2026). [Our approach to EU text provenance rules](https://openai.com/index/eu-text-provenance/).
