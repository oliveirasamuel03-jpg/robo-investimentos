# ATLAS — SEC-1: higiene de artefatos versionados em repositório público

**Data:** 12/11/2026.
**Repositório único:** `oliveirasamuel03-jpg/robo-investimentos`.
**Branch:** `atlas/sec-1-higiene-artifacts`.
**Base:** `atlas/pc-2-auditoria-seguranca-recuperacao`, SHA `7181d0a828b1ca738c9d27fba399de1f9165359f`.
**Modo:** `PAPER_ONLY / SECURITY_REVIEW / NON_OPERATIONAL_CHANGE / NO_PRODUCTION_DEPLOY`.

## 1. Resultado e decisão

**A auditoria limitada dos artefatos textuais de storage foi executada; o repositório NÃO pode ser declarado livre de segredos ou de dados pessoais.**

O repositório foi confirmado como **PÚBLICO** pela conexão GitHub. O inventário de árvore da branch auditada retornou 442 entradas, sem truncamento. Sob `storage/`, descontados os `.gitkeep`, foram identificados **61 arquivos versionados**:
- **52 arquivos textuais** (`.json`, `.csv`, `.js` e `.log`), consultados e examinados por correspondência de padrões, sem imprimir valores no relatório ou na resposta;
- **9 imagens** (`.png` e formatos equivalentes), que **não foram inspecionadas visualmente**. Imagens podem conter dados de login, nomes, e-mails, IDs, conta, URLs ou outro conteúdo sensível.

Também existem aproximadamente **137 artefatos** em `docs/retomada/evidencias/` e outros arquivos fora de storage. Estes **não fazem parte da varredura de conteúdo SEC-1**. Nenhum histórico anterior de commits/tags/PR foi inspecionado integralmente. A revisão não foi um scanner SAST/secret-scanner independente nem uma inspeção visual.

## 2. Achados por triagem heurística

No conjunto dos **52 arquivos textuais de storage** examinados:
- **Nenhum padrão típico de token de alta entropia foi identificado** pelas expressões testadas (formatos usuais sk-*, GH*, AKIA* e bloco PEM de chave privada);
- **Nenhum endereço de e-mail em formato usual foi identificado** pela expressão consultada;
- Não foi encontrado um candidato evidente de credencial de conexão em URI com query de `api_key`, `token`, `password` ou `secret` nos URLs revistos;
- Alguns arquivos incluem **URLs externas públicas** e URLs de loopback locais. URLs, isoladamente, não comprovam vazamento nem segurança.

**Importante:** falsos negativos continuam possíveis. Senhas podem ter formatos arbitrários, segredos podem estar fragmentados, criptografados, em cabeçalhos, screenshots, nomes de arquivos ou versões anteriores; padrões específicos do provedor podem não ter sido cobertos. Não reproduzir nem imprimir valor potencialmente sensível. Não afirmar que os artefatos são todos sintéticos sem prova de origem.

### Evidência de rastreabilidade — grupos de arquivos

- `storage/archive/runtime_resets/phase2_*/`: snapshots históricos de PAPER/bot, manifestos, relatórios e logs;
- `storage/tmp_real_env_validation/` e `storage/tmp_real_env_validation_hardening/`: scripts de screenshot, relatórios, logs e PNGs;
- `storage/tmp_daily_loss*/`, `storage/tmp_twelvedata_validation*/`, `storage/tmp_phase1_worker_online_validation/` e `storage/tmp_ui_audit_probe_check/`: dados e diagnósticos de validações legadas.

A varredura foi **somente leitura** desses textos; todos os conteúdos foram mantidos sem alterações. Nenhum dado foi enviado a serviços externos de segurança.

## 3. Mudanças preventivas realizadas nesta branch

### 3.1 `.gitignore`

Foram acrescentadas regras recursivas para prevenir novos rastreamentos acidentais de resultados de runtime, archives e validações sob `storage/`, preservando exceções para `.gitkeep`:

```gitignore
# Prevent future accidental commits of runtime, archive and validation outputs.
# Existing tracked historical artifacts remain tracked and require separate review.
storage/**
!storage/**/
!storage/**/.gitkeep
```

**Limites:** `.gitignore` não retira do Git arquivos **já rastreados** e não remove informações de commits antigos. Teste funcional real com `git check-ignore` e inspeção `git ls-files` permanece **PENDENTE** em ambiente com Git local; a mudança foi verificada por diff do GitHub, não por execução local de Git nesta etapa.

### 3.2 `.env.example`

O campo `ALERT_EMAIL_TO` continha um endereço pessoal fixo no exemplo público. Foi substituído por valor vazio no **arquivo de exemplo desta branch**, sem tocar `.env` real, variáveis, senhas, serviço de e-mail ou destino de alertas em produção.

**Limites:** a alteração só existe nesta branch; a `main` e o histórico antigo podem continuar exibindo o valor anterior. A troca **não representa rotação de credencial**, pois trata-se de endereço de destinatário e não de chave secreta. Nenhum histórico foi reescrito.

## 4. Classificação e riscos

| ID | Severidade | Observação | Decisão |
| --- | --- | --- | --- |
| SEC1-01 | ALTA | Repositório público com artefatos legados versionados sob `storage` | Não incluir dumps/logs/estados em PR sem triagem e autorização; mudanças de ignore apenas preventivas |
| SEC1-02 | ALTA, pendente | Nove imagens não avaliadas | Revisão visual em ambiente confiável; não presumir ausência de informações privadas |
| SEC1-03 | ALTA, pendente | Diretório de evidências de testes e histórico Git não examinados | Inventário e scanner de segredos com saída redigida; evidência de ausência parcial não equivale a certificação total |
| SEC1-04 | MÉDIA | Exemplo público incluía endereço pessoal de destino | Removido no template desta branch, sem alteração na main nem histórico |
| SEC1-05 | MÉDIA | `.gitignore` não contemplava todos os subdiretórios de resultados | Regra preventiva proposta/aplicada em branch; validar ignore com Git local |
| SEC1-06 | ALTA operacional | INFRA-1.2.6/1.2.7 e STOP sob carga continuam pendentes | SEC-1 NÃO autoriza worker, 72h, estratégia ou operação real |

Nenhum segredo comprometido foi **confirmado** nesta inspeção. Não recomendar rotação compulsória sem uma evidência concreta, nem afirmar que nenhum segredo existiu historicamente.

## 5. Testes e ações ainda necessários

**Antes de revisão de PR/merge, em executor isolado:**
1. Executar `git check-ignore -v` em caminhos fictícios, incluindo arquivos temporários, archives e `.gitkeep`, para provar o comportamento exato das regras.
2. Executar `git ls-files storage/` e classificar os rastreados; não executar `git rm` sem aprovação de preservação e tratamento histórico.
3. Fazer scanner de segredos/PII em `docs/retomada/evidencias/`, arquivos de configuração, código, histórico de branches/commits e arquivos adicionados em PR, com saída redigida e sem postar segredos em logs.
4. Revisar visualmente as nove imagens em ambiente seguro e decidir anonimização/remoção futura sem perder evidência técnica necessária.
5. Conferir que exemplos e relatórios não contêm DSNs públicos, credenciais reais, URLs de serviços sensíveis ou dados pessoais; distinguir valor de teste e credencial.
6. Se houver exposição confirmada, **não divulgar o valor**; avaliar revogação/rotação de chaves afetadas por procedimento apropriado e autorização. Reescrita histórica é uma operação sensível, fora do escopo, nunca contornar restrições GitHub.
7. Executar regressão relevante se a documentação for incorporada; preservar todos os testes financeiros existentes.
8. Revisão Atlas de segurança antes de PR e merge; nenhum PR/merge foi criado automaticamente por esta fase.

## 6. Escopo e proibições

- Não alterar main, não criar PR, não fazer merge, não operar `gh CLI` para contornar bloqueio do conector.
- Nenhuma alteração em `core/`, `pages/`, `workers/`, `paper_trading_engine.py`, `engines/`, ledger, broker, scores, thresholds, capital, ticket, provider, histórico financeiro ou conta Railway.
- Nenhum deploy, banco remoto, migração, acesso a provedor, ordem real, worker 24h, compra ou serviço pago.
- PAPER permanece obrigatório. INFRA-1.2.7 e desempenho STOP/lock são bloqueios independentes não resolvidos por esta fase.

**Status SEC-1:** mudança preventiva documental/não operacional na branch, inventário e triagem parcial de conteúdo concluídos, avaliação integral do repositório e testes de ignore pendentes. **72 horas NÃO AUTORIZADAS.**
