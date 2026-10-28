"""Cópia sem alteração do avaliador oficial do Spider. Não importe diretamente.

`exec_eval.py` faz `from parse import ...`, e `evaluation.py` faz
`from process_sql import ...` e `from exec_eval import ...`: importações
absolutas que só funcionam com a pasta do repositório original no caminho. Quem
carrega estes arquivos é `tcc_rag.evaluation.ex_oficial` (EX) e
`tcc_rag.evaluation.em_oficial` (EM e dificuldade), depois de conferir a
identidade de cada um.
Origem, licença e motivo da cópia estão em ORIGEM.md.
"""
