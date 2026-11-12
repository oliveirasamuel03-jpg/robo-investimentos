# ATLAS — BR-1: mapa de branches, dependências e gates de integração

Data: 12/11/2026. Repositório: `oliveirasamuel03-jpg/robo-investimentos`.
Branch da auditoria: `atlas/audit-2-consolidacao-seguranca-estrategia`.
**Somente inventário e planejamento; nenhuma branch foi apagada, mesclada, rebaseada ou alterada por este documento.**

## 1. Inventário de referência

A API GitHub retornou **66 branches** imediatamente antes de ser criada a branch de auditoria de consolidação, incluindo `main`. Esta branch nova aumenta o inventário para pelo menos 67, sujeito a alterações externas concorrentes. A branch principal apontava para `f0c932c21212064dcef37e1c928b0b188adf440f` na consulta. O repositório foi identificado como público.

A conexão de pesquisa de PRs não retornou PRs abertos no momento da consulta anterior. Ausência de PRs ou a propriedade `protected=false` nas branches listadas **não é autorização de merge**, nem prova de que todas as regras de proteção do repositório foram auditadas.

## 2. Linha principal auditada — NÃO É LINHA DA MAIN

```
main (f0c932c)
  └── fases INFRA-1 e anteriores em branches Atlas
      └── INFRA-1.2.3 (8dc5e5a)
          └── INFRA-1.2.4 (64ca64f)
              └── INFRA-1.2.5 (cbd9a17)
                  └── INFRA-1.2.6 (faf3fa2)
                      └── UI-1 (b7790fe)
                          └── UI-2 (c731999)
                              └── UI-3 (06ec270)
                                  └── PC-1 (6759222)
                                      └── PC-2 (7181d0a)
                                          └── SEC-1 (26fe495)
                                              └── AUDIT-2 (esta documentação)
```

Esses rótulos representam **ancestralidade de commits nas branches observadas**, não que uma fase foi aprovada para produção ou integrada à main. A INFRA-1.2.5 inclui um protótipo de snapshot rejeitado para integração; não adotá-lo silenciosamente.

## 3. Comparação factual de branches relevantes com a main

| Branch | Commits à frente da main | Commits atrás | Observação |
| --- | ---: | ---: | --- |
| INFRA-1.2.3 | 16 | 0 | Recuperação de falhas de COMMIT e STOP |
| INFRA-1.2.4 | 18 | 0 | Contenção/otimização, ainda com metas pendentes |
| INFRA-1.2.5 | 21 | 0 | Estudo de snapshot, candidato rejeitado |
| INFRA-1.2.6 | 24 | 0 | Contrato de escritores e estudo de retenção |
| UI-1 | 25 | 0 | Auditoria de interface, documentação |
| UI-2 | 26 | 0 | Especificação UX, documentação |
| UI-3 | 27 | 0 | Monitoramento móvel, documentação |
| PC-1 | 28 | 0 | Infraestrutura local e backup, documentação |
| PC-2 | 29 | 0 | Segurança/recuperação, documentação |
| SEC-1 | 32 | 0 | .gitignore, .env.example, relatório |
 
Essas contagens são do `compare_commits` no instante da inspeção; podem mudar. Vários arquivos de runtime constam no diff **acumulado da main** porque pertencem às primeiras fases INFRA. Um merge da branch SEC-1 traria também essas alterações históricas, e **não apenas os três arquivos de SEC-1**. Por isso, não recomendar merge direto da ponta da cadeia.

O conjunto anterior às novas fases ainda inclui várias branches `codex/*`, `feature/*` e `backup/*` (66 no total da lista inicial). A existência delas não comprova que estejam obsoletas: **não apagar sem comparação, PR/commit e revisão de segurança individual**.

## 4. Tabela de classificação por fase

| Grupo | Estado de engenharia | Diretriz |
| --- | --- | --- |
| INFRA-1.2.3 e 1.2.4 | Código operacional candidato com regressões históricas documentadas | Revalidar em PG16/18 e sob carga antes de integrar |
| INFRA-1.2.5 | Protótipo estudado e rejeitado para aplicação operacional | Preservar evidências; não incorporar por engano |
| INFRA-1.2.6 | Estudo + testes de contraexemplos + relatório | Exigir resultados finais de runner; não considerar P0 resolvido |
| INFRA-1.2.7 | **Não implementada** | Criar branch futura a partir de base revisada da INFRA-1.2.6; não basear automaticamente no tip da UI/SEC |
| UI-1, UI-2, UI-3 | Somente relatórios e planos | Integrar seletivamente após revisão, sem transferir acidentalmente runtime |
| PC-1, PC-2 | Documentação de host, backup e segurança | Candidatos a revisão documental |
| SEC-1 | Regras preventivas e template | Revisar .gitignore, segurança, efeitos colaterais e limpeza histórica antes de PR |
| SEC-2 + BR-1 + STRAT-1 + plano INFRA-1.2.7 | Auditorias/consolidação | Trabalho somente documental, sem autorizações de execução |

## 5. Sequência segura de revisão quando o executor voltar

1. **Não fazer merge agora.** Confirmar os SHAs e apontar explicitamente a branch-base técnica; inventariar arquivos mudados em relação a main e contra a fase anterior.
2. Executar auditoria INFRA-1.2.6 com resultados reais de teste. Em nova branch, corrigir INFRA-1.2.7 usando o último commit de INFRA-1.2.6 como ponto de partida, não incluindo automaticamente todos os documentos de interface ou alterações de SEC-1.
3. Testar a correção contra PG16/18 local sintético, restart, retenção, rollback, UNKNOWN, STOP, writer não cooperante. Se houver violação financeira, **bloquear**.
4. Avaliar separadamente o protótipo rejeitado INFRA-1.2.5 e impedir que ele altere o runtime; não incorporar ferramentas de estudo como autoridade operacional.
5. Revisar a seleção de commits documentais e as alterações preventivas de SEC-1 como mudanças próprias, evitando incorporar 32 commits de uma vez. Integração seletiva exige novo diff e prova; não há autorização automática de cherry-pick, merge, push alternativo ou PR.
6. Não criar/mesclar PR sem revisão de segurança Atlas e respeito às restrições do conector GitHub. Se o conector bloquear merge por permissão/limite/instrução, parar; jamais usar gh CLI ou workaround.
7. Não apagar branches antigas, especialmente backup ou codex, sem mapa de dependências completo e deliberação específica. Uma branch antiga pode ser a única referência de prova operacional.
8. Realizar testes de aceitação de PC/backup e UI **depois** de resolver integridade, não porque uma branch parece limpa.

## 6. Riscos críticos que o mapa NÃO resolve

- Retenção destrutiva, no-op que muda manifesto e divergência sem quarentena persistida na INFRA-1.2.6;
- STOP acima de 2 s sob volume de 5 mil eventos e contenção de lock;
- Ausência de homologação PAPER contínua, restore operacional e métricas de performance da estratégia;
- Dados legados públicos sujeitos a auditoria SEC-2 incompleta;
- Potenciais writers SQL não cooperantes sem política de privilégios no banco real.

**Status BR-1:** inventário e recomendação de integração documentados; nenhuma alteração na main, no código operacional, no broker, Railway, volumes ou serviços externos.
