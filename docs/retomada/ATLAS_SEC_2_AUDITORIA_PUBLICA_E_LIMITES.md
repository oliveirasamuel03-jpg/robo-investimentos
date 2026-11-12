# ATLAS — SEC-2: auditoria ampliada, limites da inspeção pública e plano de saneamento

Data: 12/11/2026. Repositório exclusivo: `oliveirasamuel03-jpg/robo-investimentos`.
Branch: `atlas/audit-2-consolidacao-seguranca-estrategia`.
Base: `atlas/sec-1-higiene-artifacts`, SHA `26fe4956901daa55f0c3dff58cf7678d0a7f1415`.
Escopo: **AUDITORIA_DOCUMENTAL / READ_ONLY / PAPER_ONLY / NO_PRODUCTION_DEPLOY**.
Nenhum dado financeiro real, arquivo histórico, imagem, conta, serviço ou variável foi modificado.

## 1. Veredito

**SEC-2 PARCIAL: auditoria textual ampliada, imagens e histórico completo PENDENTES.** O repositório é **público**, portanto os conteúdos versionados podem ser acessados independentemente de o robô estar desligado.

No inventário GitHub da branch SEC-1, a árvore foi retornada sem truncamento. Há **61 arquivos rastreados sob `storage/`**, excluindo `.gitkeep`: **52 textuais** (JSON/CSV/JS/log) e **9 PNG**. Os 52 textuais foram lidos na rodada anterior e/ou nesta auditoria de continuidade e submetidos a correspondências heurísticas de formatos usuais de tokens, chaves privadas, endereços eletrônicos e links de conexão, sem divulgar nenhum conteúdo suspeito no relatório.

**Resultado limitado:** nenhuma correspondência de token de API em formatos usuais ou chave privada PEM foi evidenciada nas expressões consultadas nos 52 textos; a triagem de formato de e-mail também não trouxe correspondência nesses textos. Alguns URLs externos e locais foram encontrados; a mera existência de URL não prova vazamento. Isso não equivale a detectar todas as credenciais nem garantir ausência de PII/segredos.

Uma amostra adicional de **oito arquivos textuais de `docs/retomada/evidencias/`** foi inspecionada por padrões de tokens, emails, URIs de banco e URLs. Não houve correspondências de formatos de token/e-mail/URI de banco nessa amostra; dois arquivos incluíam URLs, sem reprodução de valores. **Amostra não é censo** dos ~137 artefatos daquela pasta.

## 2. Nove imagens ainda não revisadas

Os nove PNG identificados pela árvore GitHub estão em:
- `storage/tmp_phase1_worker_online_validation/` — controle e trader (2);
- raiz `storage/` — duas capturas `tmp_real_env_post_utc_home*.png` (2);
- `storage/tmp_real_env_validation/` — controle, root, trader (3);
- `storage/tmp_real_env_validation_hardening/` — controle e trader (2).

**Tentativa de leitura:** as operações de arquivo/blob do conector GitHub recusaram conteúdo não UTF-8; a API pública de arquivos e a API pública de blobs também foram recusadas por essa limitação. O download alternativo de um PNG não foi concluído no ambiente disponível. **Nenhuma imagem foi visualmente inspecionada nesta auditoria.**

Não afirmar ausência de usuário, e-mail, IP, número de conta, token, URL ou outro dado pessoal nas capturas. **Não mover/excluir** essas imagens automaticamente; revisão visual local autenticada e preservação de evidências são necessárias.

## 3. Evidência de amostragem textual `docs/retomada/evidencias/`

Os seguintes oito arquivos foram inspecionados por padrões de texto, sem anexar seu conteúdo e sem escanear o histórico Git:
- `infra123-summary.json`;
- `infra124-summary.json`;
- `infra125-summary.json`;
- `infra123-offline.xml`;
- `infra125-postgres16.txt`;
- `infra125-checks.json`;
- `infra125-audit-before.md`;
- `banco-novo-pytest.txt`.

Nenhum dos oito indicou token de formato comum, endereço de e-mail ou URI de banco pelas expressões executadas. Dois incluíam URL. Strings codificadas/fragmentadas, arquivos maiores não inspecionados, credenciais sem prefixo conhecido e histórico anterior **não foram cobertos**.

## 4. Estado das proteções preventivas (herdado SEC-1)

- `.gitignore`: regras recursivas de `storage/**`, com exceção para diretórios e `.gitkeep`, na branch SEC-1. **Não elimina conteúdo já rastreado**, não remove acesso público histórico e o comportamento ainda exige `git check-ignore` em ambiente com Git.
- `.env.example`: o campo de destinatário `ALERT_EMAIL_TO` está vazio na branch SEC-1; mudanças anteriores em `main`/histórico não são revertidas por isso. Não se trata de token nem rotação de credencial.
- SEC-2 não adicionou outras mutações preventivas além dos documentos desta consolidação.

## 5. Ameaças e plano de decisão

| ID | Estado | Risco/ação futura |
| --- | --- | --- |
| SEC2-01 | CONFIRMADO | Projeto público com artefatos de estado e de diagnóstico versionados; revisar política de repositório e fluxo de evidências |
| SEC2-02 | PENDENTE | Revisão visual das nove PNG e classificação das capturas; não assumir que são sanitizadas |
| SEC2-03 | PENDENTE | Scanner integral de todo `docs/retomada/evidencias/`, códigos, exemplos, artefatos, branches, tags e **todo histórico Git** |
| SEC2-04 | PENDENTE | Testar `git check-ignore`, `git ls-files` e proteger reintroduções por PR/CI |
| SEC2-05 | PENDENTE | Definir política de evidências públicas com valores mascarados e arquivos privados fora do Git |
| SEC2-06 | NÃO DEMONSTRADO | Sem evidência de credencial real comprometida nos padrões inspecionados; não declarar o projeto “limpo” |
| SEC2-07 | BLOQUEADO | Qualquer ação de exclusão de arquivos, reescrita de histórico Git, push force, troca de visibilidade ou rotação de chaves requer procedimento e autorização separados |

**Procedimento recomendado na próxima janela de execução:** executar scanner de secret/PII local com saída redigida; classificar positivos por gravidade sem imprimir valores; inspecionar os nove PNG fora de repositórios compartilhados; inventariar histórico e todos os refs; confrontar resultado com evidências que precisam ser preservadas. Em caso de segredo real confirmado, tratar revogação/rotação por processo explícito e seguro, não publicar o material nem usar contornos do conector GitHub.

## 6. Escopo e status

Nenhum código operacional, branch principal, estratégia, score, threshold, min_signal_score, capital, ticket, max_open_positions, provider, broker, banco, Railway, worker, deploy, PR ou merge foi alterado. PAPER obrigatório, dinheiro real proibido. Integridade INFRA-1.2.7 e STOP sob 5.000 eventos continuam bloqueios independentes.

**SEC-2: PARCIALMENTE CONCLUÍDA E DOCUMENTADA; NÃO HOMOLOGADA COMO AUDITORIA INTEGRAL.**
